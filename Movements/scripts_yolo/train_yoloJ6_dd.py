# train_yoloJ6_debug.py
import os
import argparse
import json
from pathlib import Path
import gc
import warnings
from copy import deepcopy
import csv

# ─────────────────────────────────────────────────────────────────────────────
# Env / runtime hygiene
# ─────────────────────────────────────────────────────────────────────────────
# GUI 대기 차단 (cv2 backend waitKey 방지)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
    os.environ.pop(k, None)
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 커스텀 래퍼/IO (로컬 모듈로 임포트)
from custom_yolo import YOLO
# from custom_validator import CustomAEValidator
from custom_validator2 import CustomAEValidator
# from custom_predictor import CustomAEPredictor
from custom_predictor2 import CustomAEPredictor

# ── 디버그/차단 스위치 (환경변수)
UL_DEBUG_PRED = os.getenv("UL_DEBUG_PRED", "0") == "1"   # 디버그 프린트
NAS_OFF = os.getenv("UL_CUSTOM_PRED_NAS_OFF", "0") == "1"  # NAS 후처리 우회
AE_OFF = os.getenv("UL_AE_OFF", "0") == "1"  # AE 전처리 OFF


def _dbg(msg: str):
    if UL_DEBUG_PRED:
        print(f"[PRED-DBG] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers (추가)
# ─────────────────────────────────────────────────────────────────────────────
def _model_device(m) -> torch.device:
    """모델 파라미터가 올라간 실제 디바이스를 반환."""
    try:
        return next(m.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _as_torch_device(d) -> torch.device:
    """문자열/정수/torch.device → torch.device 정규화(선택 사용)."""
    if isinstance(d, torch.device):
        return d
    if isinstance(d, int):
        return torch.device(f"cuda:{d}") if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(d, str):
        if d.isdigit():
            return torch.device(f"cuda:{d}") if torch.cuda.is_available() else torch.device("cpu")
        return torch.device(d)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ROI(탐색 범위)
ROI_FLOAT = {
    "lr": (5e-5, 1e-3),  # log
    "weight_decay": (1e-6, 1e-3),  # log
    "fliplr": (0.0, 0.5),
    "translate": (0.0, 0.3),
    "scale_aug": (0.0, 1.0),
    "mosaic": (0.0, 0.3),
    "mixup": (0.0, 0.3),
    "cutmix": (0.0, 0.3),
    "hsv_v": (0.0, 0.5),
}
ROI_FLOAT["scale"] = ROI_FLOAT["scale_aug"]
ROI_OPTIMIZERS = ["Adam", "AdamW"]
ROI_BATCHES = [16, 24]


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_devices(devstr: str):
    s = (devstr or "").strip()
    return [d.strip() for d in s.split(",")] if s else ["0"]


def _pick_device(devlist, key: int):
    return devlist[key % len(devlist)]


def _safe_cleanup(model=None):
    try:
        del model
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()


def _auto_workers(max_default=2):
    try:
        return max(1, min(max_default, (os.cpu_count() or 8) // 2))
    except Exception:
        return 2


def _get_model_args(model):
    return getattr(model, "overrides", None) or getattr(model, "args", None) or {}


def _filtered_args(args_dict, drop_keys=("ae_infer",)):
    if args_dict is None:
        return {}
    a = dict(args_dict)
    for k in drop_keys:
        a.pop(k, None)
    return a


def _ensure_overrides(model):
    if not hasattr(model, "overrides") or model.overrides is None:
        model.overrides = {}
    model.overrides.update({"plots": True, "half": False, "amp": False})  # plots 강제 On


def _attach_custom_io(model, ae_infer=True):
    try:
        import inspect

        core = getattr(model, "model", None) or model
        base_args = _filtered_args(_get_model_args(model))
        _ensure_overrides(model)
        if not hasattr(model, "pt"):
            model.pt = True
        if not hasattr(model, "triton"):
            model.triton = False

        ch = 3
        try:
            ch = int((base_args or {}).get("channels", 3) or 3)
        except Exception:
            ch = 3
        try:
            model.ch = int(getattr(model, "ch", ch))
        except Exception:
            model.ch = ch

        for obj in (model, core):
            if not hasattr(obj, "warmup") or not callable(getattr(obj, "warmup", None)):
                setattr(obj, "warmup", lambda *a, **k: None)
            if obj is core:
                print("[DEBUG] Injected warmup() stub into DetectionModel")

        # Validator
        try:
            vkw = {}
            import inspect as _ins
            sig_v = _ins.signature(CustomAEValidator.__init__)
            if "model" in sig_v.parameters:
                vkw["model"] = core
            if "overrides" in sig_v.parameters:
                vkw["overrides"] = base_args
            elif "args" in sig_v.parameters:
                vkw["args"] = base_args
            if "_callbacks" in sig_v.parameters:
                vkw["_callbacks"] = getattr(model, "callbacks", None)
            val = CustomAEValidator(**vkw)
        except Exception:
            val = CustomAEValidator()

        try:
            if getattr(val, "stride", None) is None:
                s = getattr(core, "stride", None)
                if isinstance(s, torch.Tensor):
                    val.stride = int(s.max().item())
                elif isinstance(s, (list, tuple)):
                    val.stride = int(max(s))
                elif s is not None:
                    val.stride = int(s)
                else:
                    val.stride = 32
        except Exception:
            val.stride = 32
        model.validator = val

        # Predictor
        try:
            pkw = {}
            import inspect as _ins
            sig_p = _ins.signature(CustomAEPredictor.__init__)
            if "overrides" in sig_p.parameters:
                pkw["overrides"] = base_args
            elif "args" in sig_p.parameters:
                pkw["args"] = base_args
            if "_callbacks" in sig_p.parameters:
                pkw["_callbacks"] = getattr(model, "callbacks", None)
            pred = CustomAEPredictor(**pkw)
        except Exception:
            pred = CustomAEPredictor()

        setattr(pred, "model", model)
        try:
            if hasattr(pred, "args") and pred.args is not None and getattr(pred.args, "model", None):
                pred.args.model = None
        except Exception:
            pass
        setattr(pred, "ae_infer", bool(ae_infer))
        if not hasattr(pred, "warmup"):
            pred.warmup = lambda *a, **k: None

        model.predictor = pred
        print(f"[DEBUG] IO attached → validator={type(model.validator).__name__}, predictor={type(model.predictor).__name__}")
    except Exception as e:
        print(f"[WARN] Failed to attach custom IO: {e}")


# ▶ mAP@50 안전 추출
def _extract_map50(metrics, default=0.0) -> float:
    if metrics is None:
        return float(default)
    box = getattr(metrics, "box", None)
    if box is not None:
        v = getattr(box, "map50", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    rd = getattr(metrics, "results_dict", None)
    if isinstance(rd, dict):
        for k in ("metrics/mAP50(B)", "metrics/mAP50", "map50"):
            if k in rd:
                try:
                    return float(rd[k])
                except Exception:
                    pass
    if isinstance(metrics, dict):
        for k in ("metrics/mAP50(B)", "metrics/mAP50", "map50"):
            if k in metrics:
                try:
                    return float(metrics[k])
                except Exception:
                    pass
    return float(default)


def merge_pred_txt_to_csv(labels_dir: Path, out_csv: Path):
    rows = []
    labels_dir = Path(labels_dir)
    for p in labels_dir.rglob("*.txt"):
        stem = p.stem
        with open(p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) not in (5, 6):
                    continue
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) == 6 else None
                rows.append([stem, cls, conf, x, y, w, h])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["image_stem", "class", "conf", "x_center", "y_center", "width", "height"])
        wri.writerows(rows)
    print(f"✅ Saved predictions CSV: {out_csv}")


def dump_metrics_csv(metrics, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    maybe = {}
    if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
        maybe.update(metrics.results_dict)
    if hasattr(metrics, "box"):
        for k in ("map", "map50", "mp", "mr"):
            v = getattr(metrics.box, k, None)
            if v is not None:
                maybe[f"box/{k}"] = v
    if isinstance(metrics, dict):
        maybe.update(metrics)

    with open(out_csv, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["key", "value"])
        for k in sorted(maybe.keys()):
            v = maybe[k]
            try:
                wri.writerow([k, float(v)])
            except Exception:
                wri.writerow([k, v])
    print(f"✅ Saved test metrics CSV: {out_csv}")


def save_confusion_matrix_csv(metrics, out_csv: Path):
    """가능하면 Confusion Matrix를 CSV로도 저장(이미지 외 CSV 보조)."""
    try:
        import numpy as np
        cm_obj = getattr(metrics, "confusion_matrix", None)
        mat = getattr(cm_obj, "matrix", None)
        if mat is None and hasattr(metrics, "results_dict"):
            mat = metrics.results_dict.get("confusion_matrix", None)
        if mat is not None:
            arr = np.array(mat)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline="") as f:
                wri = csv.writer(f)
                for row in arr.tolist():
                    wri.writerow(row)
            print(f"✅ Saved confusion matrix CSV: {out_csv}")
        else:
            print("ℹ️ Confusion matrix object not found; CSV skip.")
    except Exception as e:
        print(f"⚠️ Failed to save confusion matrix CSV: {e}")


def plot_results_csv_safe(results_csv: Path, out_png: Path):
    try:
        import pandas as pd, matplotlib.pyplot as plt, matplotlib
        matplotlib.use("Agg")
    except Exception:
        print("⚠️ pandas/matplotlib 미설치로 플롯 생략.")
        return

    if not results_csv.exists():
        print(f"⚠️ results.csv 없음: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    cols = df.columns.tolist()

    keys = []
    for k in ["metrics/mAP50(B)", "metrics/mAP50", "map50"]:
        if k in cols:
            keys.append((k, "mAP50")); break
    for k in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map"]:
        if k in cols:
            keys.append((k, "mAP50-95")); break
    for k in ["train/box_loss", "box_loss", "loss/box"]:
        if k in cols:
            keys.append((k, "box_loss")); break
    for k in ["train/cls_loss", "cls_loss", "loss/cls"]:
        if k in cols:
            keys.append((k, "cls_loss")); break

    if not keys:
        print(f"⚠️ 플롯할 컬럼 없음. columns={cols}")
        return

    x = range(len(df))
    import matplotlib.pyplot as plt
    plt.figure()
    for col, label in keys:
        try:
            plt.plot(x, df[col], label=label)
        except Exception as e:
            print(f"⚠️ {col} 플롯 중 에러: {e}")
    plt.xlabel("epoch"); plt.ylabel("value"); plt.title("Learning Curves")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot: {out_png}")


# ========== 추가: 강제 저장 유틸 ==========
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_confusion_matrix_png(metrics, out_png: Path, class_names=None):
    """UL 내부 저장에 의존하지 않고 직접 CM PNG를 그려 저장."""
    try:
        import numpy as np, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm_obj = getattr(metrics, "confusion_matrix", None)
        mat = getattr(cm_obj, "matrix", None)
        if mat is None and hasattr(metrics, "results_dict"):
            mat = metrics.results_dict.get("confusion_matrix", None)
        if mat is None:
            print("ℹ️ confusion matrix not available; PNG skip.")
            return

        arr = np.array(mat, dtype=float)
        _ensure_dir(out_png.parent)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(arr, interpolation="nearest")
        plt.colorbar()
        # class names
        if class_names is None and hasattr(cm_obj, "names"):
            class_names = getattr(cm_obj, "names", None)
        if class_names is not None:
            try:
                plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
                plt.yticks(range(len(class_names)), class_names)
            except Exception:
                pass
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved confusion matrix PNG: {out_png}")
    except Exception as e:
        print(f"⚠️ Failed to save CM PNG: {e}")


def save_efficiency_matrix_png(metrics, out_png: Path):
    """
    간단한 'efficiency matrix' PNG 생성:
      [ [mAP50, mAP50-95],
        [Precision(mp), Recall(mr)] ]
    """
    try:
        import numpy as np, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        box = getattr(metrics, "box", None)
        if box is None:
            print("ℹ️ metrics.box 없음; efficiency matrix skip.")
            return
        m50 = float(getattr(box, "map50", float("nan")))
        m95 = float(getattr(box, "map", float("nan")))
        mp  = float(getattr(box, "mp", float("nan")))
        mr  = float(getattr(box, "mr", float("nan")))

        mat = np.array([[m50, m95],
                        [mp,  mr]], dtype=float)
        labels = [["mAP50", "mAP50-95"], ["Precision", "Recall"]]

        _ensure_dir(out_png.parent)
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, interpolation="nearest")
        plt.colorbar()
        # tick labels
        plt.xticks([0,1], ["Column 1", "Column 2"])
        plt.yticks([0,1], ["Row 1", "Row 2"])
        # cell text
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{labels[i][j]}\n{mat[i,j]:.3f}", ha="center", va="center")
        plt.title("Efficiency Matrix")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved efficiency matrix PNG: {out_png}")
    except Exception as e:
        print(f"⚠️ Failed to save efficiency matrix PNG: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Robust UL plot patch
# ─────────────────────────────────────────────────────────────────────────────
def _robust_plot_results(*args, **kwargs):
    import csv
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    file = kwargs.get("file", None)
    if file is None and len(args) > 0:
        file = args[0]
    if file is None:
        print("[WARN] robust_plot_results: no file provided")
        return

    file = Path(file)
    if not file.exists():
        print(f"[WARN] robust_plot_results: {file} not found")
        return

    with open(file, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows or not fieldnames:
        print("[WARN] robust_plot_results: empty results")
        return

    preferred = [
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "train/box_loss",
        "val/box_loss",
        "train/cls_loss",
        "val/cls_loss",
        "train/dfl_loss",
        "val/dfl_loss",
        "ae_loss",
        "lr/pg0", "lr/pg1", "lr/pg2"
    ]
    keys = [k for k in preferred if k in fieldnames]
    if not keys:
        print("[WARN] robust_plot_results: no known columns to plot")
        return

    if "epoch" in fieldnames:
        x = [int(r.get("epoch", "0") or 0) for r in rows]
        xlabel = "epoch"
    else:
        x = list(range(len(rows)))
        xlabel = "step"

    plt.figure()
    for k in keys:
        y = []
        for r in rows:
            v = r.get(k, "")
            try:
                import math as _m
                y.append(float(v) if v != "" else _m.nan)
            except Exception:
                y.append(float("nan"))
        plt.plot(x, y, label=k)
    plt.xlabel(xlabel)
    plt.title("training metrics")
    plt.legend()
    plt.tight_layout()
    out = file.parent / "results.png"
    plt.savefig(out)
    plt.close()
    print(f"[DEBUG] Robust results plot saved to {out}")

    try:
        from ultralytics.utils import plotting as _uplot
        _uplot.plot_results = _robust_plot_results
        print("[DEBUG] Patched ultralytics.utils.plotting.plot_results -> robust version")
    except Exception as e:
        print(f"[WARN] Could not patch plot_results: {e}")


def _patch_ul_plotting():
    """Ultralytics 내 plot_results를 미리 robust 버전으로 패치(훈련 중 사용 보장)."""
    try:
        from ultralytics.utils import plotting as _uplot
        _uplot.plot_results = _robust_plot_results
        print("[DEBUG] Pre-patched ultralytics.utils.plotting.plot_results -> robust version")
    except Exception as e:
        print(f"[WARN] Pre-patch failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# In-place loader (방법 A)
# ─────────────────────────────────────────────────────────────────────────────
def _load_inplace(model, best_pt):
    """학습에 사용한 model 인스턴스에 그대로 state_dict만 주입."""
    import torch
    print(f"[DEBUG] In-place load from: {best_pt}")
    sd = None
    try:
        ckpt = torch.load(str(best_pt), map_location="cpu")
    except Exception as e:
        print(f"[WARN] torch.load failed: {e}")
        return model

    if isinstance(ckpt, dict):
        cand = ckpt.get("ema") or ckpt.get("model") or ckpt.get("state_dict")
        if hasattr(cand, "state_dict"):
            cand = cand.state_dict()
        sd = cand

    if sd is None:
        print("[WARN] best.pt has no usable state_dict (maybe export format). Skip loading.")
        return model

    try:
        model.model.load_state_dict(sd, strict=False)
        print("[DEBUG] in-place state_dict loaded.")
    except Exception as e:
        print(f"[WARN] load_state_dict(strict=False) failed: {e}")
        return model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, fold, data_yaml, model_yaml, device, imgsz, project_root, trial_epochs=6, tune_fraction=0.6):
    lr = trial.suggest_float("lr", *ROI_FLOAT["lr"], log=True)
    wd = trial.suggest_float("weight_decay", *ROI_FLOAT["weight_decay"], log=True)
    fliplr = trial.suggest_float("fliplr", *ROI_FLOAT["fliplr"])
    translate = trial.suggest_float("translate", *ROI_FLOAT["translate"])
    scale_aug = trial.suggest_float("scale_aug", *ROI_FLOAT["scale_aug"])
    optimizer = trial.suggest_categorical("optimizer", ROI_OPTIMIZERS)
    batch = trial.suggest_categorical("batch_size", ROI_BATCHES)
    mosaic = trial.suggest_float("mosaic", *ROI_FLOAT["mosaic"])
    mixup = trial.suggest_float("mixup", *ROI_FLOAT["mixup"])
    cutmix = trial.suggest_float("cutmix", *ROI_FLOAT["cutmix"])
    hsv_v = trial.suggest_float("hsv_v", *ROI_FLOAT["hsv_v"])

    imgsz_tune = 512
    print(f"[Trial {trial.number}] fold={fold:02d} bs={batch} lr={lr:.2e} ROI-tune fraction={tune_fraction}")

    devlist = _parse_devices(device)
    device_single = _pick_device(devlist, trial.number)
    workers = _auto_workers()

    try_batch = int(batch)
    while try_batch >= 2:
        model = None
        try:
            trial_dir = project_root / f"optuna/fold{fold:02d}"
            trial_name = f"trial{trial.number:03d}"

            model = YOLO(model_yaml, task="detect", fold=fold)
            _ensure_overrides(model)
            _attach_custom_io(model, ae_infer=True)

            model.train(
                data=str(data_yaml),
                epochs=1,
                imgsz=imgsz_tune,
                batch=try_batch,
                device=device_single,
                amp=False,
                half=False,
                project=str(trial_dir),
                name=trial_name,
                lr0=lr,
                weight_decay=wd,
                optimizer=optimizer,
                patience=999,
                fliplr=fliplr,
                translate=translate,
                scale=scale_aug,
                mosaic=mosaic,
                mixup=mixup,
                cutmix=cutmix,
                hsv_v=hsv_v,
                workers=workers,
                verbose=False,
                plots=True,              # ✅ CSV/그래프 보장
                warmup_epochs=3.0,
                conf=0.001,
                iou=0.6,
                deterministic=True,
                seed=0,
                fraction=float(tune_fraction),
                save=True,
                save_period=-1,
            )

            best_map = 0.0
            for ep in range(1, int(trial_epochs)):
                model.train(
                    data=str(data_yaml),
                    epochs=1,
                    imgsz=imgsz_tune,
                    batch=try_batch,
                    device=device_single,
                    amp=False,
                    half=False,
                    project=str(trial_dir),
                    name=trial_name,
                    lr0=lr,
                    weight_decay=wd,
                    optimizer=optimizer,
                    patience=999,
                    fliplr=fliplr,
                    translate=translate,
                    scale=scale_aug,
                    mosaic=mosaic,
                    mixup=mixup,
                    cutmix=cutmix,
                    hsv_v=hsv_v,
                    workers=workers,
                    verbose=False,
                    plots=True,          # ✅
                    warmup_epochs=0.0,
                    conf=0.001,
                    iou=0.6,
                    deterministic=True,
                    seed=0,
                    fraction=float(tune_fraction),
                    resume=True,
                    save=True,
                    save_period=-1,
                )
                val_metrics = model.val(
                    data=str(data_yaml),
                    split="val",
                    conf=0.001,
                    iou=0.6,
                    batch=try_batch,
                    device=device_single,
                    verbose=False,
                    plots=True,          # ✅ 밸리데이션 그림 생성
                    validator=CustomAEValidator,
                )
                map50 = _extract_map50(val_metrics, 0.0)
                best_map = max(best_map, map50)
                trial.report(map50, step=ep)
                if trial.should_prune():
                    print(f"[PRUNED] fold={fold} trial={trial.number} ep={ep}, map50={map50:.4f}")
                    _safe_cleanup(model)
                    raise optuna.TrialPruned()

            _safe_cleanup(model)
            return float(best_map)
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] fold={fold} trial={trial.number} batch={try_batch} → halve and retry")
            _safe_cleanup(model)
            try_batch //= 2
        except optuna.TrialPruned:
            raise
        except Exception as e:
            _safe_cleanup(model)
            raise e

    raise RuntimeError("OOM: minimum batch size still OOM")


# ─────────────────────────────────────────────────────────────────────────────
# Top-K 재검증
# ─────────────────────────────────────────────────────────────────────────────
def revalidate_topk(top_trials, fold, data_yaml, model_yaml, device, imgsz, project_root, reval_epochs=20):
    devlist = _parse_devices(device)
    device_single = _pick_device(devlist, fold - 1)
    try:
        if str(device_single).isdigit():
            torch.cuda.set_device(int(device_single))
    except Exception:
        pass

    workers = max(1, _auto_workers() // 2)
    best_params, best_score = None, -1.0

    for i, t in enumerate(top_trials, 1):
        p = t.params
        print(f"[ReVal fold{fold:02d}] Top{i}/{len(top_trials)} params: {p}")
        try_batch = int(p["batch_size"])
        try_imgsz = max(384, int(imgsz * 0.9))

        while try_batch >= 2:
            model = None
            try:
                model = YOLO(model_yaml, task="detect", fold=fold)
                _ensure_overrides(model)
                model.overrides["amp"] = False
                _attach_custom_io(model, ae_infer=True)

                model.train(
                    data=str(data_yaml),
                    epochs=int(reval_epochs),
                    imgsz=try_imgsz,
                    batch=try_batch,
                    device=device_single,
                    project=str(project_root / f"optuna_reval/fold{fold:02d}"),
                    name=f"top{i:02d}",
                    lr0=float(p["lr"]),
                    optimizer=p["optimizer"],
                    weight_decay=float(p["weight_decay"]),
                    patience=max(20, int(reval_epochs // 3)),
                    amp=False,
                    half=False,
                    fliplr=float(p["fliplr"]),
                    translate=float(p["translate"]),
                    scale=float(p.get("scale_aug", p.get("scale", 0.5))),
                    mosaic=float(p["mosaic"]),
                    mixup=float(p["mixup"]),
                    cutmix=float(p["cutmix"]),
                    hsv_v=float(p["hsv_v"]),
                    workers=workers,
                    verbose=False,
                    plots=True,      # ✅
                    warmup_epochs=3.0,
                    conf=0.001,
                    iou=0.6,
                    deterministic=True,
                    seed=0,
                )

                metrics = model.val(
                    data=str(data_yaml),
                    split="val",
                    conf=0.001,
                    iou=0.6,
                    batch=max(2, try_batch // 2),
                    device=device_single,
                    verbose=False,
                    plots=True,      # ✅
                    validator=CustomAEValidator,
                )
                map50 = _extract_map50(metrics, 0.0)
                print(f"[ReVal fold{fold:02d}] Top{i} mAP50={map50:.5f}")
                if map50 > best_score:
                    best_score, best_params = map50, deepcopy(p)
                break

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM ReVal] fold={fold} Top{i} batch={try_batch}, imgsz={try_imgsz} → 축소 재시도")
                _safe_cleanup(model)
                try_batch = max(2, try_batch // 2)
                if try_imgsz > 320:
                    try_imgsz = max(256, int(try_imgsz * 0.9))
            except RuntimeError as e:
                msg = str(e)
                if "expandable_segment" in msg or "CUDA out of memory" in msg:
                    print(f"[OOM-like ReVal] {msg.splitlines()[0]} → 축소 재시도")
                    _safe_cleanup(model)
                    try_batch = max(2, try_batch // 2)
                    if try_imgsz > 320:
                        try_imgsz = max(320, int(try_imgsz * 0.9))
                else:
                    _safe_cleanup(model)
                    raise
            finally:
                _safe_cleanup(model)

        if try_batch < 2:
            print(f"[ReVal fold{fold:02d}] Top{i} 중단: 최소 배치에서도 OOM")
            continue

    return best_params, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from types import MethodType
    from ultralytics.models.yolo.detect import DetectionPredictor as _BasePred

    # UL plot 함수 사전 패치(훈련 중 결과 이미지 보장)
    _patch_ul_plotting()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_yaml", type=str, required=True)
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--intensity", type=int, required=True)

    parser.add_argument("--auto_optuna", action="store_true")
    parser.add_argument("--optuna_trials1", type=int, default=32)
    parser.add_argument("--optuna_trial_epochs", type=int, default=10)
    parser.add_argument("--optuna_topk", type=int, default=5)
    parser.add_argument("--optuna_reval_epochs", type=int, default=25)
    parser.add_argument("--tune_fraction", type=float, default=1.0)

    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale_aug", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=20)    # fast 기본값
    parser.add_argument("--imgsz", type=int, default=512)    # fast 기본값(고정)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--test_plots", action="store_true")

    args = parser.parse_args()

    yaml_p = Path(args.model_yaml)
    assert yaml_p.exists(), f"YAML 파일 없음: {yaml_p}"
    IMGSZ = int(args.imgsz)
    EPOCHS = int(args.epochs)
    ACCUM = max(1, int(args.accumulate))
    PROJECT = Path("runs") / yaml_p.stem
    PROJECT.mkdir(exist_ok=True, parents=True)

    devlist = _parse_devices(args.device)
    workers = _auto_workers()

    for fold in range(1,6):
        fs = f"fold{fold:02d}"
        base = Path(f"../data/UMay/yolo_gas/yolo_split_{fs}")
        data_y = base / "data_aug_1var_5_gs.yaml"
        test = base / "images" / "test"
        savej = (base / "labels" / "test").exists()

        # 1) Optuna (옵션)
        if args.auto_optuna:
            print(f"\n🔍 {fs} Optuna tuning (ROI) trials={args.optuna_trials1}, trial_epochs={args.optuna_trial_epochs}")
            pruner = MedianPruner(n_warmup_steps=2, interval_steps=1)
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42, n_startup_trials=4),
                pruner=pruner
            )

            best_json_path = PROJECT / f"{fs}_best.json"
            if best_json_path.exists():
                try:
                    with open(best_json_path, "r") as f:
                        hint = json.load(f)
                    allowed = {
                        "lr","weight_decay","fliplr","translate","scale_aug",
                        "optimizer","batch_size","mosaic","mixup","cutmix","hsv_v",
                    }
                    hint = {k: hint[k] for k in hint.keys() & allowed}
                    for k, (lo, hi) in ROI_FLOAT.items():
                        if k in hint:
                            hint[k] = _clamp(hint[k], lo, hi)
                    if "optimizer" in hint and hint["optimizer"] not in ROI_OPTIMIZERS:
                        hint["optimizer"] = "Adam"
                    if "batch_size" in hint and hint["batch_size"] not in ROI_BATCHES:
                        hint["batch_size"] = min(ROI_BATCHES, key=lambda x: abs(int(hint["batch_size"]) - x))

                    study.enqueue_trial(hint)
                    print(f"[Optuna] Enqueued seed trial for {fs}: {hint}")
                except Exception as e:
                    print(f"[Optuna] WARNING: enqueue seed failed: {e}")

            study.optimize(
                lambda t: objective(
                    t, fold, data_y, str(yaml_p), args.device, IMGSZ, PROJECT,
                    trial_epochs=int(args.optuna_trial_epochs),
                    tune_fraction=float(args.tune_fraction)
                ),
                n_trials=int(args.optuna_trials1)
            )

            topk = min(args.optuna_topk, len(study.best_trials))
            top_trials = sorted(
                study.best_trials,
                key=lambda tr: tr.value if tr.value is not None else -1.0,
                reverse=True
            )[:topk]

            best_params, best_score = revalidate_topk(
                top_trials, fold, data_y, str(yaml_p), args.device, IMGSZ, PROJECT,
                reval_epochs=int(args.optuna_reval_epochs)
            )
            if best_params is None:
                best_params = study.best_params

            print(f"[{fs}] Final best params: {best_params}")
            (PROJECT / f"{fs}_best.json").write_text(json.dumps(best_params, indent=2))

            lr, wd = best_params["lr"], best_params["weight_decay"]
            fl, tr = best_params["fliplr"], best_params["translate"]
            sa, opti, bs = best_params["scale_aug"], best_params["optimizer"], best_params["batch_size"]
            mo, mu, cm, hv = best_params["mosaic"], best_params["mixup"], best_params["cutmix"], best_params["hsv_v"]
        else:
            lr, wd = 1e-3, 5e-4
            fl, tr = args.fliplr, args.translate
            sa, opti, bs = args.scale_aug, "Adam", 16
            mo, mu, cm, hv = args.mosaic, 0.0, 0.0, 0.0

        # 2) 본 학습
        device_train = _pick_device(devlist, fold - 1)
        nbs = ACCUM * bs

        model = YOLO(str(yaml_p), task="detect", fold=fold)
        _ensure_overrides(model)
        _attach_custom_io(model, ae_infer=(not AE_OFF))  # AE 전처리 스위치 반영

        model.train(
            data=str(data_y),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=bs,
            device=device_train,
            project=str(PROJECT / "train" / fs),
            name="run",
            lr0=lr,
            optimizer=opti,
            weight_decay=wd,
            patience=10,
            save_period=5,
            amp=False,
            half=False,
            val=True,
            fliplr=fl,
            translate=tr,
            scale=sa,
            mosaic=mo,
            mixup=mu,
            cutmix=cm,
            hsv_v=hv,
            workers=workers,
            verbose=True,
            plots=True,            # ✅ 학습 CSV/결과 이미지 생성 보장
            warmup_epochs=3.0,
            conf=0.001,
            iou=0.6,
            deterministic=True,
            seed=0,
            nbs=nbs,
        )

        # ✅ 실제 저장 폴더(run/run2/...)를 trainer에서 가져와 사용
        train_dir = Path(getattr(getattr(model, "trainer", None), "save_dir",
                                 PROJECT / "train" / fs / "run"))
        results_csv = train_dir / "results.csv"
        _robust_plot_results(file=results_csv)
        plot_results_csv_safe(results_csv, train_dir / "efficiency.png")

        # 3) 테스트 & 예측
        # ✅ 가중치도 train_dir 기준으로 탐색
        weights_dir = train_dir / "weights"
        best_pt = None
        for fname in ("best_ae.pt", "best.pt"):
            p = weights_dir / fname
            if p.exists():
                best_pt = p
                break

        if best_pt is not None:
            model = _load_inplace(model, best_pt)
            _attach_custom_io(model, ae_infer=(not AE_OFF))

            v_impl = type(getattr(model, "validator", None)).__name__ if getattr(model, "validator", None) else "None"
            p_impl = type(getattr(model, "predictor", None)).__name__ if getattr(model, "predictor", None) else "None"
            print(f"[DEBUG] Validator Impl: {v_impl} | Predictor Impl: {p_impl}")

            # ✅ 중요: 이전 학습 Validator 끊기 → 테스트용 save_dir이 test 쪽으로 생성되게
            try:
                model.validator = None
            except Exception:
                pass

            metrics = model.val(
                data=str(data_y),
                split="test",
                batch=bs,
                device=device_train,
                project=str(PROJECT / "test" / fs),
                name="run",
                save_json=savej,
                plots=True,          # ✅ Confusion Matrix/PR/ROC 등 생성 시도
                verbose=True,
                conf=0.001,
                iou=0.6,
                validator=CustomAEValidator,
            )

            # ✅ 실제 테스트 저장 폴더(run/run2/...)를 validator에서 가져옴
            test_dir = Path(getattr(getattr(model, "validator", None), "save_dir",
                                    PROJECT / "test" / fs / "run"))
            _ensure_dir(test_dir)

            # 메트릭 & CM CSV/PNG 강제 저장 (UL 내부 저장과 독립)
            dump_metrics_csv(metrics, test_dir / "test_metrics.csv")
            save_confusion_matrix_csv(metrics, test_dir / "confusion_matrix.csv")
            save_confusion_matrix_png(metrics, test_dir / "confusion_matrix.png")
            save_efficiency_matrix_png(metrics, test_dir / "efficiency_matrix.png")

            # ── 추론: 리스트 입력 + workers=0
            img_globs = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.png")
            img_list = []
            for g in img_globs:
                img_list += [str(p) for p in Path(test).glob(g)]
            img_list = sorted(img_list)[:32]
            print(f"[DEBUG] Predict files: {len(img_list)} (showing 3) -> {img_list[:3]}")
            assert img_list, f"No images found under: {test}"

            # ─────────────────────────────────────────────────────────────
            # SMOKE TEST 1: 모델 forward만 1회
            # ─────────────────────────────────────────────────────────────
            try:
                pr = getattr(model, "predictor", None)
                if pr is not None and hasattr(pr, "setup_model"):
                    pr.setup_model(model)
                core = getattr(model, "model", None) or model
                core.eval()
                dev = _model_device(core)
                import time
                x = torch.zeros(1, 3, IMGSZ, IMGSZ, device=dev, dtype=torch.float32)
                t0 = time.time()
                with torch.inference_mode():
                    y = core(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                print(f"[SMOKE] forward ok in {time.time()-t0:.3f}s; y type={type(y)} dev={dev}")
            except Exception as e:
                print("[SMOKE] forward FAILED:", e)
                raise

            # ─────────────────────────────────────────────────────────────
            # SMOKE TEST 2: AE 전처리만 1회
            # ─────────────────────────────────────────────────────────────
            try:
                pr = getattr(model, "predictor", None)
                if pr is not None:
                    pr.setup_model(model)
                core = getattr(model, "model", None) or model
                dev = _model_device(core)
                im = torch.rand(1, 3, IMGSZ, IMGSZ, device=dev)
                import time
                t0 = time.time()
                out = pr.preprocess(im)  # AE 있으면 여기서 AE forward
                if torch.cuda.is_available():
                    torch.cuda.synchronize(dev)
                print(f"[SMOKE] preprocess ok in {time.time()-t0:.3f}s; out.shape={tuple(out.shape)} dev={dev}")
            except Exception as e:
                print("[SMOKE] preprocess FAILED:", e)
                raise

            # ─────────────────────────────────────────────────────────────
            # NAS 후처리 우회 스위치 적용 (원형 postprocess로 대체)
            # ─────────────────────────────────────────────────────────────
            if NAS_OFF and getattr(model, "predictor", None) is not None:
                try:
                    from ultralytics.models.yolo.detect import DetectionPredictor as _BasePred
                    from types import MethodType
                    model.predictor.postprocess = MethodType(_BasePred.postprocess, model.predictor)
                    print("[DEBUG] NAS-like postprocess is OFF (fallback to base).")
                except Exception as e:
                    print(f"[WARN] disable NAS-like postprocess failed: {e}")

            # AE 전처리 스위치(환경변수) 반영
            if getattr(model, "predictor", None) is not None:
                model.predictor.ae_infer = (not AE_OFF)

            _dbg("predict: start")
            pred = model.predict(
                source=img_list,
                imgsz=IMGSZ,
                device=device_train,
                batch=16,
                workers=0,
                project=str(PROJECT / "predict" / fs),
                name="run",
                save=True,
                save_txt=True,
                save_conf=True,
                verbose=True,
                stream_buffer=False,
                conf=0.25,
                iou=0.6,
                show=False,        # GUI 차단
                visualize=False,   # 시각화 hook 차단
            )
            _dbg("predict: end")

            # ✅ 실제 예측 저장 폴더(run/run2/...)를 predictor에서 가져옴
            try:
                pred_dir = Path(model.predictor.save_dir)
            except Exception:
                pred_dir = Path(PROJECT / "predict" / fs / "run")
            labels_dir = pred_dir / "labels"
            if labels_dir.exists():
                merge_pred_txt_to_csv(labels_dir, pred_dir / "predictions.csv")

            map50 = _extract_map50(metrics, 0.0)
            print(f"[{fs}] Test mAP50 = {map50:.4f}")
        else:
            print(f"[{fs}] best.pt 없음, 스킵")

        _safe_cleanup(model)


if __name__ == "__main__":
    main()
