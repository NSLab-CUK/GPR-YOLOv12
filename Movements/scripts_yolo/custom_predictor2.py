# custom_predictor.py
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


class CustomAEPredictor(DetectionPredictor):
    # 8.3.x: __init__(overrides, _callbacks) 시그니처
    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        # 안전 모드
        if hasattr(self, "args") and self.args is not None:
            self.args.half = False
            self.args.amp = False
            # ❗ YAML 경로가 남아있으면 AutoBackend가 파일로 로딩 시도 → 비워둔다
            try:
                if getattr(self.args, "model", None):
                    self.args.model = None
            except Exception:
                pass
        self.autoencoder = None  # setup_model에서 연결

    def _unwrap_core(self, m):
        """YOLO 래퍼(model) → 내부 DetectionModel(nn.Module)로 풀기."""
        if m is None:
            return None
        core = getattr(m, "model", None)
        return core if isinstance(core, nn.Module) else (m if isinstance(m, nn.Module) else None)

    def setup_model(self, model=None):
        """
        Predictor가 받는 model 인자를 우선 사용하고(None이면 self.model 사용),
        YOLO 래퍼라면 내부 nn.Module(DetectionModel)로 풀어서 AutoBackend에 넘긴다.
        또한 args.model(YAML 경로)을 비워 파일 로딩을 차단한다.
        """
        # 1) model 해석 (None → self.model, 래퍼 → core nn.Module)
        m = model if model is not None else getattr(self, "model", None)
        core = self._unwrap_core(m)

        # 2) YAML 로딩 차단
        if hasattr(self, "args") and self.args is not None:
            try:
                if getattr(self.args, "model", None):
                    self.args.model = None
            except Exception:
                pass

        # 3) 상위에 nn.Module을 넘기면 AutoBackend가 in-memory 모델로 셋업
        super().setup_model(core if core is not None else m)

        # 4) warmup 없는 버전 대비
        m2 = getattr(self, "model", None)
        if m2 is not None and not hasattr(m2, "warmup"):
            setattr(m2, "warmup", lambda *a, **k: None)

        # 5) AE 연결 및 FP32 강제
        #   - self.model가 래퍼면 래퍼→core 순으로 autoencoder 검색
        ae = None
        if m is not None and getattr(m, "autoencoder", None) is not None:
            ae = m.autoencoder
        if ae is None and core is not None and getattr(core, "autoencoder", None) is not None:
            ae = core.autoencoder

        if ae is not None:
            self.autoencoder = ae
            self.autoencoder.float()
        # Predictor 내부 모델도 FP32
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.float()
            except Exception:
                pass

        # 6) ✅ Predictor가 기대하는 속성 강제 주입
        #    - AutoBackend/DetectionModel 어느 쪽이더라도 fp16 속성 보장
        if hasattr(self, "model") and not hasattr(self.model, "fp16"):
            try:
                setattr(self.model, "fp16", False)
            except Exception:
                pass
        #    - self.device 보장
        if not hasattr(self, "device") or self.device is None:
            try:
                self.device = next(self.model.parameters()).device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def preprocess(self, im):
        # ⛑️ UL preprocess가 self.model.fp16에 접근하므로 여기서도 1회 더 보장
        try:
            if not hasattr(self.model, "fp16"):
                setattr(self.model, "fp16", False)
        except Exception:
            pass
        if not hasattr(self, "device") or self.device is None:
            try:
                self.device = next(self.model.parameters()).device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        im = super().preprocess(im)             # [B,C,H,W], [0,1]
        im = im.to(self.device).float().clamp(0.0, 1.0)
        ae = getattr(self.model, "autoencoder", None) or self.autoencoder
        if ae is not None:
            with torch.no_grad():
                out = ae(im)
                out = out[0] if isinstance(out, tuple) else out
                out = out.float().clamp(0.0, 1.0)
                if out.ndim != 4:
                    raise RuntimeError(f"[CustomAEPredictor] AE output must be 4D, got {out.shape}")
                if out.shape[1] == 3:
                    return out
                i_1ch = im[:, 0:1] if im.shape[1] > 1 else im
                diff = (i_1ch - out).abs()
                return torch.cat([i_1ch, out, diff], dim=1)
        return im

    # ─────────────────────────────────────────────────────────────────────────
    # NAS 스타일 원시 출력 후처리 지원 (xyxy→xywh + concat(scores)) → 상위 postprocess
    # ─────────────────────────────────────────────────────────────────────────
    def postprocess(self, preds_in, img, orig_imgs):
        """
        NAS 계열 출력처럼 [bboxes_xyxy, class_scores] 또는 [[bboxes_xyxy, class_scores]] 형태를 자동 인식하여
        xywh로 변환 후 상위 postprocess에 맞는 텐서로 변환한다.
        - bboxes: [B, N, 4] (xyxy)
        - scores: [B, N, C]
        변환 후:
        - preds:  [B, 4+C, N]  (UL postprocess 기대 형식)
        다른 포맷이면 안전하게 super().postprocess로 폴백한다.
        """
        try:
            bboxes, scores = None, None

            # Case A) preds_in = [bboxes, scores]
            if isinstance(preds_in, (list, tuple)) and len(preds_in) >= 2 and \
               torch.is_tensor(preds_in[0]) and torch.is_tensor(preds_in[1]):
                bboxes, scores = preds_in[0], preds_in[1]

            # Case B) preds_in = [[bboxes, scores]] (배치 래핑)
            elif isinstance(preds_in, (list, tuple)) and len(preds_in) == 1:
                inner = preds_in[0]
                if isinstance(inner, (list, tuple)) and len(inner) >= 2 and \
                   torch.is_tensor(inner[0]) and torch.is_tensor(inner[1]):
                    bboxes, scores = inner[0], inner[1]

            # bboxes/scores가 발견되면 NAS 스타일로 변환
            if bboxes is not None and scores is not None:
                # 배치 차원 보정: [N,4] / [N,C] → [1,N,4] / [1,N,C]
                if bboxes.ndim == 2:
                    bboxes = bboxes.unsqueeze(0)
                if scores.ndim == 2:
                    scores = scores.unsqueeze(0)

                # xyxy → xywh
                boxes_xywh = ops.xyxy2xywh(bboxes)

                # [B,N,4] + [B,N,C] → [B,N,4+C] → [B,4+C,N]
                preds = torch.cat((boxes_xywh, scores), dim=-1).permute(0, 2, 1).contiguous()

                return super().postprocess(preds, img, orig_imgs)

        except Exception as e:
            print(f"[CustomAEPredictor] postprocess(NAS-like) fallback due to: {e}")

        # 인식 불가 포맷이면 원형 경로로 처리
        return super().postprocess(preds_in, img, orig_imgs)
