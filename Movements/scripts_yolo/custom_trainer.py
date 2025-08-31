# custom_trainer.py
import os
import torch
import torch.nn.functional as F
from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.losses_ae import hybrid_loss
from ultralytics.nn.modules.ae import AutoEncoder
from custom_validator import CustomAEValidator


class CustomAETrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("✅ CustomAETrainer initialized!")

    def _setup_train(self, world_size=1):
        super()._setup_train(world_size)
        # ▶ 안전망: 학습 시작 전에 FP32로 고정
        self.model.float()
        if hasattr(self, "ema") and self.ema and getattr(self.ema, "ema", None) is not None:
            self.ema.ema.float()
        if hasattr(self.model, "autoencoder") and self.model.autoencoder is not None:
            self.model.autoencoder.float()

    def train_batch(self, batch):
        # 안전하게 FP32로 통일
        imgs   = batch["img"].to(self.device).float()
        labels = batch["labels"]

        # ── AutoEncoder forward (FP32) ───────────────────────────────
        ae_out = self.model.autoencoder(imgs)  # AE는 FP32 권장
        if isinstance(ae_out, tuple):
            ae_out = ae_out[0]

        # 기대: [B,3,H,W] in [0,1]
        ae_out = ae_out.clamp(0, 1).float().to(self.device)
        assert ae_out.ndim == 4 and ae_out.shape[1] == 3, "AE output must be [B,3,H,W]"
        imgs_ae = ae_out

        # ── Detection loss (Ultralytics 표준 경로) ────────────────────
        det_loss, loss_items = super().train_batch({"img": imgs_ae, "labels": labels})

        # ── AE reconstruction loss ───────────────────────────────────
        # 채널 분리 (i, i_hat, diff) 가정
        i_norm  = imgs_ae[:, 0:1]
        i_hat_n = imgs_ae[:, 1:2]

        if hasattr(self.model.autoencoder, "get_reconstruction"):
            i_hat_for_loss = self.model.autoencoder.get_reconstruction(imgs).clamp(0, 1).float()
        else:
            i_hat_for_loss = i_hat_n

        recon_loss = hybrid_loss(i_hat_for_loss, i_norm)
        lambda_ae  = float(getattr(self.args, "lambda_ae", 0.01))
        total_loss = det_loss + lambda_ae * recon_loss

        self.log_dict({"train/ae_loss": float(recon_loss)})

        return total_loss, loss_items

    def get_validator(self):
        val_split = self.data.get("val") or self.data.get("test")
        dataloader = self.get_dataloader(val_split, batch_size=self.batch_size, rank=-1, mode='val')

        v_args = copy(self.args)
        # ▶ 검증 인자에서도 확실히 OFF
        v_args.half = False
        v_args.amp = False

        return CustomAEValidator(
            model=self.model,
            dataloader=dataloader,
            save_dir=self.save_dir,
            args=v_args,
            _callbacks=self.callbacks,
        )

    # custom_trainer.py (CustomAETrainer 내부)
    def final_eval(self):
        # 검증용 모델(EMA 우선) 확보
        m = self.ema.ema if (
                    hasattr(self, "ema") and self.ema and getattr(self.ema, "ema", None) is not None) else self.model

        # AE 없으면 재부착
        if (not hasattr(m, "autoencoder")) or (getattr(m, "autoencoder", None) is None):
            if hasattr(self.model, "autoencoder") and self.model.autoencoder is not None:
                m.autoencoder = self.model.autoencoder
                print("[DEBUG FINAL EVAL] Re-attached AE onto eval model.")

        # FP32 강제
        m.float()
        if hasattr(m, "autoencoder") and m.autoencoder is not None:
            m.autoencoder.float()

        # validator를 trainer 기반으로 호출 → validator(self)
        metrics = self.validator(self)
        self.metrics = metrics
        if hasattr(self, "fitness_key") and self.fitness_key in metrics:
            self.fitness = metrics[self.fitness_key]
        return self.metrics

    def load(self, weights):
        print(f"[DEBUG] Loading YOLO weights from: {weights}")
        ckpt = torch.load(weights, map_location="cpu")
        model_dict = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(model_dict, strict=False)
        print("[DEBUG] YOLO backbone weights loaded.")

        # 🔥 AE 재부착 + FP32 강제
        self.model.autoencoder = AutoEncoder(loss_type="hybrid").to(self.device).float()

        ae_weights = getattr(self.model, "ae_weights", None)
        if ae_weights is not None and hasattr(ae_weights, "exists") and ae_weights.exists():
            self.model.autoencoder.load_state_dict(torch.load(ae_weights, map_location="cpu"))
            print(f"[DEBUG] AE weights loaded from: {ae_weights}")
        else:
            print("[WARN] AE weights not found or path is None.")

        # 전체 모델 FP32 고정
        self.model.float()
        return self
