import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.losses_ae import hybrid_loss
from pathlib import Path
import yaml
import inspect


def yaml_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_split_dir(src, data_dict, args):
    """
    src(train/val/test)이 상대경로면 data['path']와 args.data(YAML 경로), CWD를 후보 루트로 삼아
    '존재하는' 경로를 우선 선택해 절대경로로 반환. 이미 절대경로면 그대로 반환.
    """
    if isinstance(src, (list, tuple)) and src:
        src = src[0]
    if not src:
        return None
    sp = Path(src)
    if sp.is_absolute():
        return sp.as_posix()

    yaml_dir = None
    ad = getattr(args, "data", None)
    if isinstance(ad, (str, Path)):
        try:
            yaml_dir = Path(ad).resolve().parent
        except Exception:
            yaml_dir = Path(ad).parent if isinstance(ad, (str, Path)) else None

    base_raw = ""
    if isinstance(data_dict, dict):
        base_raw = data_dict.get("path", "") or ""

    bases = []
    if base_raw:
        brp = Path(base_raw)
        if brp.is_absolute():
            bases.append(brp)
        else:
            if yaml_dir is not None:
                bases.append((yaml_dir / brp).resolve())
            bases.append((Path.cwd() / brp).resolve())
    if yaml_dir is not None:
        bases.append(yaml_dir.resolve())
    bases.append(Path.cwd().resolve())

    uniq_bases, seen = [], set()
    for b in bases:
        try:
            key = b.as_posix()
        except Exception:
            continue
        if key not in seen:
            uniq_bases.append(b)
            seen.add(key)

    tried = []
    for root in uniq_bases:
        c = (root / sp).resolve()
        tried.append(c.as_posix())
        if c.exists():
            return c.as_posix()

    if uniq_bases:
        fallback = (uniq_bases[0] / sp).resolve().as_posix()
        print(f"[WARN] split dir does not exist, using fallback: {fallback}\n tried: {tried}")
        return fallback

    fb = (Path.cwd() / sp).resolve().as_posix()
    print(f"[WARN] split dir resolution failed, defaulting to: {fb}")
    return fb


class CustomAEValidator(DetectionValidator):
    def __init__(self, model=None, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

        # 안전 모드 + 플롯 강제 ON
        if hasattr(self, "args") and self.args is not None:
            self.args.half = False
            self.args.amp = False
            self.args.plots = True  # 반드시 True

        # 모델/AE 핸들
        m = model if isinstance(model, nn.Module) else getattr(self, "model", None)
        if m is not None:
            self.model = m
        self.autoencoder = getattr(m, "autoencoder", None) if m is not None else None
        self.lambda_ae = float(getattr(self.args, "lambda_ae", 1.0)) if hasattr(self, "args") and self.args else 1.0
        self.ae_loss_fn = hybrid_loss

        # split 기본값
        if not hasattr(self, "args") or self.args is None:
            class _A: pass
            self.args = _A()
            self.args.split = "val"
            self.args.plots = True
        elif not hasattr(self.args, "split"):
            self.args.split = "val"

        # data 적재
        if getattr(self, "data", None) is None:
            ad = getattr(self.args, "data", None)
            if isinstance(ad, (str, Path)):
                self.data = yaml_load(ad)
            else:
                self.data = ad
        if isinstance(getattr(self, "data", None), dict):
            self.data.setdefault("channels", 3)

        # 저장 디렉토리 선 확보
        try:
            if getattr(self, "save_dir", None):
                Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[WARN] could not ensure save_dir: {e}")

    def _get_names(self):
        names = getattr(self, "names", None)
        if names is None and isinstance(self.data, dict):
            names = self.data.get("names", None)
        return names

    def _force_save_plots(self, metrics=None):
        """가능하면 Confusion Matrix 이미지를 강제로 생성(UL 내부 흐름과 별개로)."""
        sd = Path(getattr(self, "save_dir", "."))
        try:
            sd.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            cm = getattr(self, "confusion_matrix", None)
            if cm is not None and hasattr(cm, "plot"):
                cm.plot(save_dir=sd, names=self._get_names())
                p = sd / "confusion_matrix.png"
                print(f"✅ Confusion matrix image forced: {p}")
            else:
                print("ℹ️ No confusion_matrix on validator; CM force-plot skipped.")
        except Exception as e:
            print(f"⚠️ Force CM plot failed: {e}")

    def _resolve_split_with_key(self):
        want = getattr(self.args, "split", "val")
        d = getattr(self, "data", None)
        if not isinstance(d, dict):
            return None, "val"
        for k in (want, "val", "test", "train"):
            raw = d.get(k, None)
            resolved = _resolve_split_dir(raw, d, self.args)
            if resolved:
                return resolved, k
        return None, "val"

    def __call__(self, *args, **kwargs):
        # trainer 해석
        trainer = kwargs.pop("trainer", None)
        if trainer is None and len(args) > 0:
            trainer = args[0]

        # model 해석
        m = kwargs.pop("model", None)
        m = m if isinstance(m, nn.Module) else None

        if trainer is not None:
            ema_obj = getattr(trainer, "ema", None)
            if getattr(ema_obj, "ema", None) is not None and isinstance(ema_obj.ema, nn.Module):
                m = ema_obj.ema
            elif isinstance(getattr(trainer, "model", None), nn.Module):
                m = trainer.model

        if m is None:
            m = getattr(self, "model", None)

        if not isinstance(m, nn.Module):
            raise RuntimeError("[CustomAEValidator] No valid nn.Module model available for validation.")

        # stride 보정
        if getattr(self, "stride", None) is None:
            try:
                s = getattr(m, "stride", None)
                if isinstance(s, torch.Tensor):
                    self.stride = int(s.max().item())
                elif isinstance(s, (list, tuple)):
                    self.stride = int(max(s))
                elif s is not None:
                    self.stride = int(s)
                else:
                    self.stride = 32
            except Exception:
                self.stride = 32

        # dataloader 구성
        if getattr(self, "dataloader", None) is None:
            bs = int(getattr(self.args, "batch", 16))

            if trainer is not None:
                want = getattr(self.args, "split", "val")
                split_key = None
                for k in (want, "val", "test", "train"):
                    if getattr(trainer, "data", {}) and trainer.data.get(k, None) is not None:
                        split_key = k
                        break
                if split_key is None:
                    raise RuntimeError("[CustomAEValidator] split path not found in trainer.data")

                raw_src = trainer.data[split_key]
                split_src = _resolve_split_dir(raw_src, getattr(trainer, "data", None) or self.data, self.args)
                print(f"[DEBUG] {split_key} dir resolved to: {split_src}")

                td = trainer.get_dataloader
                sig = inspect.signature(td)
                kw = {}
                if "batch" in sig.parameters:
                    kw["batch"] = getattr(trainer, "batch_size", bs) or bs
                if "batch_size" in sig.parameters:
                    kw["batch_size"] = getattr(trainer, "batch_size", bs) or bs
                if "mode" in sig.parameters:
                    kw["mode"] = "test" if split_key == "test" else ("train" if split_key == "train" else "val")
                if "rank" in sig.parameters:
                    kw["rank"] = -1
                self.dataloader = td(split_src, **kw)
            else:
                split_src, split_key = self._resolve_split_with_key()
                if split_src is None:
                    ad = getattr(self.args, "data", None)
                    if isinstance(ad, (str, Path)):
                        self.data = yaml_load(ad)
                        self.data.setdefault("channels", 3)
                    split_src, split_key = self._resolve_split_with_key()
                if split_src is None:
                    raise RuntimeError("[CustomAEValidator] Could not resolve dataset split path.")

                gd = self.get_dataloader
                sig = inspect.signature(gd)
                kw = {}
                if "batch" in sig.parameters:
                    kw["batch"] = bs
                if "batch_size" in sig.parameters:
                    kw["batch_size"] = bs
                if "mode" in sig.parameters:
                    kw["mode"] = "test" if split_key == "test" else ("train" if split_key == "train" else "val")
                if "rank" in sig.parameters:
                    kw["rank"] = -1
                print(f"[DEBUG] {split_key} dir resolved to: {split_src}")
                self.dataloader = gd(split_src, **kw)

        # AE 재부착 + FP32 강제
        if (not hasattr(m, "autoencoder")) or (getattr(m, "autoencoder", None) is None):
            if getattr(self, "autoencoder", None) is not None:
                m.autoencoder = self.autoencoder
                print("[DEBUG] Re-attached AutoEncoder onto model for validation.")

        if hasattr(self, "args"):
            self.args.half = False
            self.args.amp = False
            self.args.plots = True
        m.float()
        if getattr(m, "autoencoder", None) is not None:
            m.autoencoder.float()

        # 모델 dtype/device 기록
        try:
            p = next(m.parameters())
            self._val_device = p.device
            self._val_dtype = p.dtype
        except StopIteration:
            self._val_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._val_dtype = torch.float32

        # 상위 호출 → 결과 반환 전에 강제 플롯 저장
        out = super().__call__(trainer, model=m, **kwargs) if trainer is not None else super().__call__(model=m, **kwargs)
        try:
            self._force_save_plots(out)
        except Exception as e:
            print(f"[WARN] force-save-plots failed: {e}")
        return out

    def preprocess(self, batch):
        batch = super().preprocess(batch)

        imgs = batch["img"].to(self._val_device).float().clamp(0.0, 1.0)

        if getattr(self, "autoencoder", None) is not None:
            with torch.no_grad():
                i_1ch = imgs[:, 0:1] if imgs.shape[1] > 1 else imgs
                out = self.autoencoder(i_1ch)
                out = out[0] if isinstance(out, tuple) else out
                out = out.float().clamp(0.0, 1.0)

            if out.ndim != 4:
                raise RuntimeError(f"[CustomAEValidator] AE output must be 4D [B,C,H,W], got {out.shape}")

            recon = out[:, 1:2] if out.shape[1] == 3 else out
            diff = (i_1ch - recon).abs()
            imgs_ae = torch.cat([i_1ch, recon, diff], dim=1).to(self._val_dtype)
            batch["img"] = imgs_ae
            batch["ae_loss"] = float(self.lambda_ae) * self.ae_loss_fn(recon, i_1ch)
        else:
            batch["img"] = imgs.to(self._val_dtype)

        return batch

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        if "ae_loss" in batch:
            self.ae_losses = getattr(self, "ae_losses", [])
            self.ae_losses.append(float(batch["ae_loss"]))

    def get_stats(self):
        stats = super().get_stats()
        if getattr(self, "ae_losses", None):
            try:
                stats["ae_loss"] = sum(self.ae_losses) / len(self.ae_losses)
            except Exception:
                pass
        return stats
