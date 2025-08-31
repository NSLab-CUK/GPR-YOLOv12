#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# train_yolo_J7_no.py — OOM 내성 적용판 (AE 미사용, 순정 Ultralytics YOLO만 사용)
# - best.json(폴드별) 존재 시: 탐색범위 자동 축소 + 시드 트라이얼 enqueue
# - Optuna/ReVal/본학습 OOM 세이프 처리
# - ⏱ 시간 단축 패치: trials/epochs/topk 축소 + tune_fraction + 튜닝 해상도 축소

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
for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
    os.environ.pop(k, None)

# ✅ 조각화 완화 옵션
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128,garbage_collection_threshold:0.8"
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ★ AE 경로 차단: 순정 YOLO만 사용
from ultralytics import YOLO


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

def _auto_workers(max_default=4):
    try:
        return max(1, min(max_default, (os.cpu_count() or 8) // 2))
    except Exception:
        return 2

def _clip(a, lo, hi):
    return max(lo, min(hi, a))

def _range_around(c, width, lo, hi):
    return (_clip(c - width, lo, hi), _clip(c + width, lo, hi))

def _log_span(c, mul_lo, mul_hi, lo, hi):
    return (_clip(c * mul_lo, lo, hi), _clip(c * mul_hi, lo, hi))

def _load_fold_hint(project_root: Path, fold: int):
    """runs/<yaml_stem>/foldXX_best.json 이 있으면 로드해서 반환."""
    fs = f"fold{fold:02d}"
    p = project_root / f"{fs}_best.json"
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            hint = json.load(f)
        allowed = {
            "lr","weight_decay","fliplr","translate","scale_aug",
            "optimizer","batch_size","mosaic","mixup","cutmix","hsv_v",
        }
        return {k: hint[k] for k in hint.keys() & allowed}
    except Exception:
        return None

# ⬇️ 다버전 호환: DetMetrics / results_dict / dict 모두 지원
def _extract_map50(metrics, default=0.0) -> float:
    """Ultralytics val 결과에서 mAP@50을 안전하게 추출한다."""
    if metrics is None:
        return float(default)

    box = getattr(metrics, "box", None)
    if box is not None:
        val = getattr(box, "map50", None)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
        rd = getattr(metrics, "results_dict", None)
        if isinstance(rd, dict) and "metrics/mAP50(B)" in rd:
            try:
                return float(rd["metrics/mAP50(B)"])
            except Exception:
                pass

    if isinstance(metrics, dict):
        for k in ("metrics/mAP50(B)", "metrics/mAP50", "map50"):
            if k in metrics:
                try:
                    return float(metrics[k])
                except Exception:
                    return float(default)

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

def plot_efficiency_from_results(results_csv: Path, out_png: Path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        print("⚠️ pandas/matplotlib 미설치로 efficiency plot 생략(훈련 plots=True면 results.png 생성).")
        return

    if not results_csv.exists():
        print(f"⚠️ results.csv 없음: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    epoch = df.index.values

    plt.figure()
    key = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else ("metrics/mAP50" if "metrics/mAP50" in df.columns else None)
    if key:
        plt.plot(epoch, df[key], label="mAP50")
    if "train/box_loss" in df.columns:
        plt.plot(epoch, df["train/box_loss"], label="box_loss")
    if "train/cls_loss" in df.columns:
        plt.plot(epoch, df["train/cls_loss"], label="cls_loss")
    plt.xlabel("epoch"); plt.ylabel("value"); plt.title("Efficiency (learning curves)")
    plt.legend(); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved efficiency plot: {out_png}")

# ─────────────────────────────────────────────────────────────────────────────
# 안전 검증: conf 낮춰 재평가(mAP=0 완화)
# ─────────────────────────────────────────────────────────────────────────────
def _val_with(model_or_path, **kwargs):
    if isinstance(model_or_path, YOLO):
        return model_or_path.val(**kwargs)
    return YOLO(model_or_path).val(**kwargs)

def safe_validate(model_or_path, data_yaml, imgsz, device, batch=None, base_conf=0.001, iou=0.6, verbose=False):
    """1차(conf=0.001) → 저조 시 2차(conf=1e-4, agnostic_nms=True) 평가."""
    def once(conf, agnostic=False):
        return _val_with(
            model_or_path,
            data=str(data_yaml),
            split="val",
            conf=conf,
            iou=iou,
            batch=batch,
            device=device,
            verbose=verbose,
            plots=False,
            agnostic_nms=agnostic,
        )

    m1 = once(base_conf, agnostic=False)
    map50_1 = _extract_map50(m1, 0.0)
    if map50_1 and map50_1 > 1e-6:
        return m1, map50_1

    m2 = once(1e-4, agnostic=True)
    map50_2 = _extract_map50(m2, 0.0)
    return m2, map50_2


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective (per-epoch 진행 + pruner 지원) - AE 비사용
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, fold, data_yaml, model_yaml, device, imgsz, project_root,
              trial_epochs=12, tune_fraction=1.0):

    # 기본 절대 한계 (안전 클립)
    ABS = {
        "lr": (1e-5, 3e-2),
        "weight_decay": (1e-7, 1e-3),
        "fliplr": (0.0, 0.5),
        "translate": (0.0, 0.3),
        "scale_aug": (0.0, 0.7),
        "mosaic": (0.0, 0.6),
        "mixup": (0.0, 0.3),
        "cutmix": (0.0, 0.3),
        "hsv_v": (0.0, 0.5),
    }

    # fold별 best.json 있으면 그 주변으로 범위 축소
    hint = _load_fold_hint(Path(project_root), fold)

    if hint:
        # 로그스케일(±2x), 연속값(대략 ±폭)으로 집중
        lr_lo, lr_hi = _log_span(float(hint.get("lr", 8e-4)), 0.5, 2.0, *ABS["lr"])
        wd_lo, wd_hi = _log_span(float(hint.get("weight_decay", 1.6e-5)), 0.5, 2.0, *ABS["weight_decay"])
        fl_lo, fl_hi = _range_around(float(hint.get("fliplr", 0.14)), 0.08, *ABS["fliplr"])
        tr_lo, tr_hi = _range_around(float(hint.get("translate", 0.16)), 0.05, *ABS["translate"])
        sc_lo, sc_hi = _range_around(float(hint.get("scale_aug", 0.10)), 0.08, *ABS["scale_aug"])
        mo_lo, mo_hi = _range_around(float(hint.get("mosaic", 0.29)), 0.10, *ABS["mosaic"])
        mu_lo, mu_hi = _range_around(float(hint.get("mixup", 0.10)), 0.04, *ABS["mixup"])
        cm_lo, cm_hi = _range_around(float(hint.get("cutmix", 0.066)), 0.04, *ABS["cutmix"])
        hv_lo, hv_hi = _range_around(float(hint.get("hsv_v", 0.096)), 0.07, *ABS["hsv_v"])

        lr          = trial.suggest_float("lr",           lr_lo, lr_hi, log=True)
        wd          = trial.suggest_float("weight_decay", wd_lo, wd_hi, log=True)
        fliplr      = trial.suggest_float("fliplr",       fl_lo, fl_hi)
        translate   = trial.suggest_float("translate",    tr_lo, tr_hi)
        scale_aug   = trial.suggest_float("scale_aug",    sc_lo, sc_hi)
        mosaic      = trial.suggest_float("mosaic",       mo_lo, mo_hi)
        mixup       = trial.suggest_float("mixup",        mu_lo, mu_hi)
        cutmix      = trial.suggest_float("cutmix",       cm_lo, cm_hi)
        hsv_v       = trial.suggest_float("hsv_v",        hv_lo, hv_hi)
        optimizer   = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        batch       = trial.suggest_categorical("batch_size", [16, 24])

    else:
        # 넓은 기본 범위
        lr          = trial.suggest_float("lr",           5e-5, 1e-3, log=True)
        wd          = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        fliplr      = trial.suggest_float("fliplr",       0.0, 0.5)
        translate   = trial.suggest_float("translate",    0.0, 0.3)
        scale_aug   = trial.suggest_float("scale_aug",    0.0, 0.5)
        mosaic      = trial.suggest_float("mosaic",       0.0, 0.4)
        mixup       = trial.suggest_float("mixup",        0.0, 0.3)
        cutmix      = trial.suggest_float("cutmix",       0.0, 0.3)
        hsv_v       = trial.suggest_float("hsv_v",        0.0, 0.5)
        optimizer   = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        batch       = trial.suggest_categorical("batch_size", [16, 24])

    # ⏱ 튜닝 전용 해상도 0.75× (최소 320)
    imgsz_tune = max(320, int(imgsz * 0.75))

    trial_epochs_eff = max(5, int(trial_epochs))
    print(f"[Trial {trial.number}] fold={fold:02d} batch={batch} lr={lr:.2e} trial_epochs={trial_epochs_eff} tune_fraction={tune_fraction}")

    devlist = _parse_devices(device)
    device_single = _pick_device(devlist, trial.number)
    workers = _auto_workers()

    try_batch = int(batch)
    while try_batch >= 2:
        model = None
        try:
            trial_dir = Path(project_root) / f"optuna/fold{fold:02d}"
            trial_name = f"trial{trial.number:03d}"

            model = YOLO(model_yaml)

            # 첫 1ep (워밍업 포함) — ⏱ imgsz 축소 + fraction 사용
            model.train(
                data=str(data_yaml),
                epochs=1,
                imgsz=imgsz_tune,
                batch=try_batch,
                device=device_single,
                amp=False, half=False,
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
                plots=False,
                warmup_epochs=3.0,
                conf=0.001, iou=0.6,
                deterministic=True, seed=0,
                fraction=float(tune_fraction),
                save=True, save_period=-1,
                cos_lr=True,
            )

            best_map = 0.0
            for ep in range(1, int(trial_epochs_eff)):
                model.train(
                    data=str(data_yaml),
                    epochs=1,
                    imgsz=imgsz_tune,
                    batch=try_batch,
                    device=device_single,
                    amp=False, half=False,
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
                    plots=False,
                    warmup_epochs=0.0,
                    conf=0.001, iou=0.6,
                    deterministic=True, seed=0,
                    fraction=float(tune_fraction),
                    resume=True,
                    save=True, save_period=-1,
                    cos_lr=True,
                )

                val_metrics, map50 = safe_validate(
                    model, data_yaml=data_yaml, imgsz=imgsz_tune, device=device_single,
                    batch=try_batch, base_conf=0.001, iou=0.6, verbose=False
                )
                best_map = max(best_map, map50)
                trial.report(map50, step=ep)

                if trial.should_prune():
                    print(f"[PRUNED] fold={fold} trial={trial.number} at epoch={ep}, map50={map50:.4f}")
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
# Top-K 재검증 — ✅ OOM-세이프(배치 반감 + imgsz 축소) + AMP ON + workers 축소 + set_device 고정
# ─────────────────────────────────────────────────────────────────────────────
def revalidate_topk(top_trials, fold, data_yaml, model_yaml, device, imgsz, project_root, reval_epochs=40):
    devlist = _parse_devices(device)
    device_single = _pick_device(devlist, fold - 1)

    # 디바이스 고정 (다른 프로세스 기본 0 번 문제 완화)
    try:
        if str(device_single).isdigit():
            torch.cuda.set_device(int(device_single))
    except Exception:
        pass

    workers = max(1, _auto_workers() // 2)  # 메모리 여유를 위해 축소
    best_params, best_score = None, -1.0

    for i, t in enumerate(top_trials, 1):
        p = t.params
        print(f"[ReVal fold{fold:02d}] Top{i}/{len(top_trials)} params: {p}")

        try_batch = int(p["batch_size"])
        # ⏱ 재검증에서도 약간 축소 (최소 384)
        try_imgsz = max(384, int(imgsz * 0.9))

        while try_batch >= 2:
            model = None
            try:
                model = YOLO(model_yaml)

                # ── 학습 (AMP ON, workers↓)
                model.train(
                    data=str(data_yaml),
                    epochs=int(reval_epochs),
                    imgsz=try_imgsz,
                    batch=try_batch,
                    device=device_single,
                    project=str(Path(project_root) / f"optuna_reval/fold{fold:02d}"),
                    name=f"top{i:02d}",
                    lr0=float(p["lr"]),
                    optimizer=p["optimizer"],
                    weight_decay=float(p["weight_decay"]),
                    patience=max(20, int(reval_epochs // 3)),
                    amp=True, half=False,        # ✅ AMP로 메모리 절감
                    fliplr=float(p["fliplr"]),
                    translate=float(p["translate"]),
                    scale=float(p["scale_aug"]),
                    mosaic=float(p["mosaic"]),
                    mixup=float(p["mixup"]),
                    cutmix=float(p["cutmix"]),
                    hsv_v=float(p["hsv_v"]),
                    workers=workers,
                    verbose=False,
                    plots=False,
                    warmup_epochs=3.0,
                    conf=0.001, iou=0.6,
                    deterministic=True, seed=0,
                    cos_lr=True,
                )

                # ── 검증 (배치는 절반으로 안전하게)
                metrics = model.val(
                    data=str(data_yaml),
                    split="val",
                    conf=0.001,
                    iou=0.6,
                    batch=max(2, try_batch // 2),
                    device=device_single,
                    verbose=False,
                    plots=False,
                )

                map50 = _extract_map50(metrics, 0.0)
                print(f"[ReVal fold{fold:02d}] Top{i} mAP50={map50:.5f}")
                if map50 > best_score:
                    best_score, best_params = map50, deepcopy(p)
                break  # 성공 → 루프 탈출

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM ReVal] fold={fold} Top{i} batch={try_batch}, imgsz={try_imgsz} → 축소 재시도")
                _safe_cleanup(model)
                try_batch = max(2, try_batch // 2)
                if try_imgsz > 320:
                    try_imgsz = max(320, int(try_imgsz * 0.9))  # 해상도도 점진 축소
            except Exception as e:
                _safe_cleanup(model)
                raise e
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_yaml", type=str, required=True)
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--n", type=int, required=True)          # (인터페이스 유지용)
    parser.add_argument("--intensity", type=int, required=True)  # (인터페이스 유지용)

    # ⏱ Optuna 기본값을 절반 수준으로 단축
    parser.add_argument("--auto_optuna", action="store_true")
    parser.add_argument("--optuna_trials1", type=int, default=18)         # 36 → 18
    parser.add_argument("--optuna_trial_epochs", type=int, default=6)     # 12 → 6
    parser.add_argument("--optuna_topk", type=int, default=3)             # 5  → 3
    parser.add_argument("--optuna_reval_epochs", type=int, default=20)    # 40 → 20
    parser.add_argument("--tune_fraction", type=float, default=1.0)       # 1.0 → 0.6

    # 수동 기본 증강/학습 파라미터
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale_aug", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)

    # 본 학습
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=512)
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

    for fold in range(2, 11):
        fs = f"fold{fold:02d}"
        base = Path(f"../data/UMay/yolo_gas/yolo_split_{fs}")
        data_y = base / "data_aug_1var_5_gs.yaml"

        # test split 사용
        test = base / "images" / "test"
        savej = (base / "labels" / "test").exists()

        # 1) Optuna 튠
        if args.auto_optuna:
            print(f"\n🔍 {fs} Optuna tuning (trials={args.optuna_trials1}, trial_epochs={args.optuna_trial_epochs})")
            # ⏱ 가속된 Pruner/Sampler
            pruner = MedianPruner(n_warmup_steps=2, interval_steps=1)  # 3→2
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42, n_startup_trials=4),       # startup 축소
                pruner=pruner
            )

            # 기존 best.json이 있으면 시드 트라이얼 enqueue
            hint = _load_fold_hint(PROJECT, fold)
            if hint:
                if hint.get("optimizer") not in ["Adam", "AdamW", "SGD"]:
                    hint["optimizer"] = "Adam"
                if int(hint.get("batch_size", 24)) not in [8, 16, 24, 32, 48]:
                    cands = [8, 16, 24, 32, 48]
                    bs = int(hint.get("batch_size", 24))
                    hint["batch_size"] = min(cands, key=lambda x: abs(x - bs))
                study.enqueue_trial(hint)
                print(f"[Optuna] Enqueued seed trial for {fs}: {hint}")

            study.optimize(
                lambda t: objective(t, fold, data_y, str(yaml_p), args.device, IMGSZ, PROJECT,
                                    trial_epochs=int(args.optuna_trial_epochs),
                                    tune_fraction=float(args.tune_fraction)),
                n_trials=int(args.optuna_trials1)
            )

            topk = min(args.optuna_topk, len(study.best_trials))
            top_trials = sorted(
                study.best_trials,
                key=lambda tr: tr.value if tr.value is not None else -1.0,
                reverse=True
            )[:topk]

            best_params, _ = revalidate_topk(
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

        # 2) 본 학습 (OOM-세이프: OOM 시 배치 반감 후 재시도)
        device_train = _pick_device(devlist, fold - 1)
        try_batch_main = int(bs)
        nbs = ACCUM * try_batch_main

        while try_batch_main >= 2:
            model = None
            try:
                model = YOLO(str(yaml_p))  # ★ AE 없음
                model.train(
                    data=str(data_y),
                    epochs=EPOCHS,
                    imgsz=IMGSZ,
                    batch=try_batch_main,
                    device=device_train,
                    project=str(PROJECT / "train" / fs),
                    name="run",
                    lr0=lr,
                    optimizer=opti,
                    weight_decay=wd,
                    patience=20,
                    save_period=5,
                    amp=False,                 # 필요 시 True로 바꿔 메모리 절감 가능
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
                    plots=True,                # results.csv, curves
                    warmup_epochs=3.0,
                    conf=0.001,
                    iou=0.6,
                    deterministic=True,
                    seed=0,
                    nbs=ACCUM * try_batch_main,  # 내부 accumulate 고려
                    cos_lr=True,
                )
                break  # 학습 성공
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM Main] fold={fold} train batch={try_batch_main} → halve and retry")
                _safe_cleanup(model)
                try_batch_main //= 2

        if try_batch_main < 2:
            print(f"[{fs}] 본 학습 중단: 최소 배치에서도 OOM")
            continue

        # 3) 테스트 & 예측 (테스트도 OOM-세이프)
        weights_dir = PROJECT / "train" / fs / "run" / "weights"
        best_pt = None
        for fname in ("best.pt",):
            p = weights_dir / fname
            if p.exists():
                best_pt = p
                break

        if best_pt is not None:
            model = YOLO(str(best_pt))  # 안전하게 가중치 재로딩

            try_batch_eval = try_batch_main
            while try_batch_eval >= 2:
                try:
                    metrics = model.val(
                        data=str(data_y),
                        split="test",
                        batch=try_batch_eval,
                        device=device_train,
                        project=str(PROJECT / "test" / fs),
                        name="run",
                        save_json=savej,
                        plots=True,
                        verbose=True,
                        conf=0.001,
                        iou=0.6,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"[OOM Test] fold={fold} val batch={try_batch_eval} → halve and retry")
                    _safe_cleanup(None)
                    try_batch_eval //= 2

            dump_metrics_csv(metrics, Path(PROJECT / "test" / fs / "run" / "test_metrics.csv"))
            results_csv = PROJECT / "train" / fs / "run" / "results.csv"
            plot_efficiency_from_results(results_csv, PROJECT / "train" / fs / "run" / "efficiency.png")

            # 예측도 배치 축소 재시도
            try_batch_pred = min(16, try_batch_eval)
            while try_batch_pred >= 2:
                try:
                    pred = model.predict(
                        source=str(test),
                        device=device_train,
                        batch=try_batch_pred,
                        project=str(PROJECT / "predict" / fs),
                        name="run",
                        save=True,
                        save_txt=True,
                        save_conf=True
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"[OOM Predict] fold={fold} pred batch={try_batch_pred} → halve and retry")
                    _safe_cleanup(None)
                    try_batch_pred //= 2

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
