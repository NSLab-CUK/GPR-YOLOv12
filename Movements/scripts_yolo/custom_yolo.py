from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from custom_trainer import CustomAETrainer
from custom_validator import CustomAEValidator
# from custom_predictor import CustomAEPredictor
from custom_predictor2 import CustomAEPredictor  # ← 현재 프로젝트에서 사용하는 Predictor

import ultralytics.nn.tasks
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.ae import AutoEncoder
from ultralytics.nn.modules.losses_ae import hybrid_loss
from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)

from ultralytics.utils import ROOT, YAML
from ultralytics.nn.modules.attention import CBAM, SEBlock, ChannelAttention, SpatialAttention
import ultralytics.nn.tasks as tasks

# YAML 파서가 참조하는 tasks.py 전역에 등록
tasks.CBAM = CBAM
tasks.SEBlock = SEBlock
tasks.ChannelAttention = ChannelAttention
tasks.SpatialAttention = SpatialAttention

ROOT_DIR = Path(__file__).resolve().parent.parent


def _ensure_device_consistency(self_model: nn.Module):
    """
    모델과 모델에 부착된 AE의 device/dtype을 일관화한다.
    - 기준은 self_model(DetectionModel)의 첫 번째 파라미터.
    - AE가 존재하면 동일 device/dtype으로 이동하고, eval 상태를 정렬.
    """
    tgt: nn.Module = getattr(self_model, "module", self_model)
    # 기준 파라미터
    try:
        p = next(tgt.parameters())
        dev, dt = p.device, p.dtype
    except StopIteration:
        dev, dt = (torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float32)

    # AE 정렬
    ae = getattr(tgt, "autoencoder", None)
    if ae is not None:
        try:
            ae.to(device=dev, dtype=dt)
        except Exception:
            # 일부 PyTorch 버전에서 dtype 인자를 무시하는 경우가 있어 2단계로 보정
            ae.to(dev)
            try:
                for mp in ae.parameters():
                    mp.data = mp.data.to(dt)
            except Exception:
                pass
        if not tgt.training:
            ae.eval()


class YOLO(Model):
    def __init__(self, model: Union[str, Path], task=None, verbose=False, fold=None):
        path = Path(model if isinstance(model, (str, Path)) else "")
        print(f"[DEBUG] 모델 로딩: path={path}, stem={path.stem}, suffix={path.suffix}")

        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__

        elif "ae" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            print("[DEBUG] YOLOEAutoEncoder 로딩")
            new_instance = YOLOEAutoEncoder(path, task=task, verbose=verbose, fold=fold)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            super().__init__(model=model, task=task, verbose=verbose)
            from ultralytics.models.yolo.yoloe.head import YOLOEDetect
            print("[DEBUG] 모델 마지막 레이어:", type(self.model.model[-1]))
            print("[DEBUG] YOLOEDetect 인지 여부:", isinstance(self.model.model[-1], YOLOEDetect))

            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():
                from ultralytics import RTDETR
                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    def __init__(self, model: Union[str, Path] = "yolov8s-world.pt", verbose: bool = False) -> None:
        super().__init__(model=model, task="detect", verbose=verbose)
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            "detect": {
                "model": WorldModel,
                "trainer": yolo.world.WorldTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }

    def set_classes(self, classes: List[str]) -> None:
        self.model.set_classes(classes)
        if " " in classes:
            classes.remove(" ")
        self.model.names = classes
        if self.predictor:
            self.predictor.model.names = classes


class YOLOEAutoEncoder(Model):
    def __init__(
        self,
        model: Union[str, Path] = "yoloe.yaml",
        task: Optional[str] = None,
        verbose: bool = False,
        fold: Optional[int] = None,
        ae_loss_fn=None,
    ):
        print("[DEBUG] YOLOEAutoEncoder 모델 초기화")
        super().__init__(model=model, task=task, verbose=verbose)

        # --- 1) AE 생성 및 (선택) 가중치 로드 -------------------------
        ae = AutoEncoder(loss_type="hybrid")
        self.ae_loss_type = "hybrid"
        self.ae_weights: Optional[Path] = None

        if fold is not None:
            ae_path = ROOT_DIR / "AE" / "pretrained_folds6" / f"pretrained_ae_fold{fold:02d}.pth"
            self.ae_weights = ae_path
            if ae_path.exists():
                print(f"🧠 AE weights loaded from: {ae_path}")
                ae.load_state_dict(torch.load(ae_path, map_location="cpu"), strict=False)
            else:
                print(f"[WARN] AE weights not found at {ae_path}")

        # --- 2) DDP/DP 안전 부착: model.module 우선 --------------------
        tgt: nn.Module = getattr(self.model, "module", self.model)
        tgt.autoencoder = ae
        self.autoencoder = ae  # (선택) 래퍼에도 보관

        # --- 3) 모델 파라미터 기준으로 장치/정밀도 정렬 ----------------
        _ensure_device_consistency(self.model)

        # --- 4) 기타 설정 ----------------------------------------------
        self.ae_loss_fn = ae_loss_fn or hybrid_loss

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load((ROOT / "cfg/datasets/coco8.yaml"))["names"]

    @property
    def task_map(self):
        return {
            "detect": {
                "model": YOLOEModel,
                "trainer": CustomAETrainer,
                "validator": CustomAEValidator,
                "predictor": CustomAEPredictor,  # ← 커스텀 Predictor 사용 (custom_predictor2)
            },
            "segment": {
                "model": YOLOESegModel,
                "trainer": yolo.yoloe.YOLOESegTrainer,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
        }

    # (선택) 모델 내부 forward에서 AE를 통과시켜 사용하려는 경우 유지
    def forward(self, x, *args, **kwargs):
        print(f"[DEBUG AE→YOLOE] forward 호출: x.shape={tuple(x.shape)}")
        tgt = getattr(self.model, "module", self.model)
        ae = getattr(tgt, "autoencoder", None)
        if ae is not None:
            # AE 파라미터 기준 dtype/device로 입력 정렬
            try:
                p = next(ae.parameters())
                x = x.to(device=p.device, dtype=p.dtype, non_blocking=True)
            except StopIteration:
                pass

            # AE 구간은 fp32 유지(autocast 비활성) → dtype 충돌 예방
            from torch.cuda.amp import autocast
            with autocast(enabled=False):
                x_ae = ae(x)
                x_ae = x_ae[0] if isinstance(x_ae, tuple) else x_ae

            if getattr(getattr(self, "overrides", {}), "verbose", False):
                print(f"[DEBUG AE→YOLOE] AE 출력: x_ae.shape={tuple(x_ae.shape)}, mean={x_ae.mean().item():.4f}")
            return self.model(x_ae, *args, **kwargs)
        return self.model(x, *args, **kwargs)

    def compute_ae_loss(self, x):
        tgt = getattr(self.model, "module", self.model)
        ae = getattr(tgt, "autoencoder", None)
        if ae is None:
            raise RuntimeError("[YOLOEAutoEncoder.compute_ae_loss] AutoEncoder not attached to model.")

        # AE 파라미터 기준 dtype/device로 입력 정렬
        try:
            p = next(ae.parameters())
            x = x.to(device=p.device, dtype=p.dtype, non_blocking=True)
        except StopIteration:
            pass

        with torch.no_grad():
            x_hat = ae.get_reconstruction(x)
        return hybrid_loss(x_hat, x)

    def val(self, validator=None, **kwargs):
        # ── 0) eval 모드 고정 ──────────────────────────────────────────
        self.model.eval()
        tgt: nn.Module = getattr(self.model, "module", self.model)
        ae = getattr(tgt, "autoencoder", None)
        if ae is not None:
            ae.eval()
            print("[DEBUG] AutoEncoder loaded and set to eval()")
        else:
            print("[INFO] AE not found on wrapper; validator will re-attach if needed.")

        # ── 1) args 구성 ───────────────────────────────────────────────
        data_path = kwargs.pop("data", None) or str(
            ROOT_DIR / "data" / "UMay" / "yolo_gas" / "yolo_split_fold01" / "data_aug_1var_5_gs.yaml"
        )
        args = {
            **getattr(self, "overrides", {}),
            **kwargs,
            "mode": "val",
            "data": data_path,
        }
        args["plots"] = False  # SciPy 이슈 방지

        print(f"[DEBUG] validator args keys: {list(args.keys())}")
        print(f"[DEBUG] validator args['data']: {args['data']}")

        # ── 2) validator 클래스 선택 ───────────────────────────────────
        validator_cls = validator or self._smart_load("validator")
        print(f"[DEBUG] validator class: {validator_cls}")

        # ── 3) trainer 필요 ────────────────────────────────────────────
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            raise RuntimeError("[YOLO.val] Trainer is None. Call after training or attach a trainer before validation.")

        # 기존 validator 재사용(없으면 생성)
        v = getattr(trainer, "validator", None)
        if v is None or not isinstance(v, validator_cls):
            v = validator_cls(
                model=self.model,        # EMA 선택은 CustomAEValidator가 내부 처리
                dataloader=None,         # 아래에서 생성
                save_dir=getattr(trainer, "save_dir", None),
                args=args,
                _callbacks=self.callbacks,
            )
            trainer.validator = v

        # ── 4) dataloader 없으면 생성 (rank=-1로 단일 프로세스 검증) ───
        if getattr(v, "dataloader", None) is None:
            split = args.get("split", "val")
            val_split = trainer.data.get(split) or trainer.data.get("val") or trainer.data.get("test")
            v.dataloader = trainer.get_dataloader(val_split, batch_size=trainer.batch_size, rank=-1, mode="val")

        # ── 5) 검증 실행 ──────────────────────────────────────────────
        metrics = v(trainer)  # CustomAEValidator.__call__에 FP32/AE 재부착/EMA 선택 포함

        # ── 6) 결과 저장 후 반환 ──────────────────────────────────────
        self.metrics = metrics
        return metrics

    def load(self, weights):
        print(f"[DEBUG] Loading YOLO weights from: {weights}")
        ckpt = torch.load(weights, map_location="cpu")

        model_dict = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(model_dict, strict=False)
        print("[DEBUG] YOLO backbone weights loaded.")

        # --- AE 다시 생성 및 로드 (DDP 안전 부착) ----------------------
        print("[DEBUG] Reattaching AutoEncoder...")
        ae = AutoEncoder(loss_type=self.ae_loss_type)

        if self.ae_weights is not None and Path(self.ae_weights).exists():
            ae.load_state_dict(torch.load(self.ae_weights, map_location="cpu"), strict=False)
            print(f"[DEBUG] AE weights loaded from: {self.ae_weights}")
        else:
            print("[WARN] AE weights not found or path is None.")

        tgt: nn.Module = getattr(self.model, "module", self.model)
        tgt.autoencoder = ae
        self.autoencoder = ae  # 래퍼에도 다시 노출

        # 모델 파라미터 기준 장치/정밀도 정렬 + eval
        _ensure_device_consistency(self.model)
        return self
