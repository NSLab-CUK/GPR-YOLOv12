# ===============================
# File: ae/train_all_folds.py  (pretraining loop)
# ===============================
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics.nn.modules.ae import AutoEncoder  # ← 위 파일 경로에 맞춰 import
from gpr_dataset import GPRDataset
from pathlib import Path
import yaml

# ─────────────────────────────
# 기본 설정
# ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 고정 해상도면 성능 ↑

BASE_DIR = Path("../data/UMay/yolo_gas")
SAVE_DIR = Path("./pretrained_folds6")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3

INTENSITY = 1
N_AUG = 5

# ─────────────────────────────
# PSNR (per-image 평균, data_range 파라미터화)
# ─────────────────────────────
def compute_psnr(img1, img2, data_range=1.0, eps=1e-10):
    """
    img1, img2: (N, 1, H, W). 값 범위가 [0,1]이면 data_range=1.0, [0,255]면 255.0
    배치 내 각 샘플의 PSNR을 계산하여 평균을 반환.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"PSNR expects same shapes, got {tuple(img1.shape)} vs {tuple(img2.shape)}")
    mse = F.mse_loss(img1, img2, reduction='none')   # (N,1,H,W)
    mse = mse.flatten(1).mean(dim=1)                 # (N,)
    psnr = 20 * torch.log10(torch.as_tensor(data_range, device=img1.device)) \
           - 10 * torch.log10(mse.clamp_min(eps))
    return psnr.mean()

# ─────────────────────────────
# 학습 루프
# ─────────────────────────────
for fold in range(1, 11):
    fold_name = f"fold{fold:02d}"
    yaml_path = BASE_DIR / f"yolo_split_{fold_name}" / "data_aug_1var_5_gs.yaml"

    if not yaml_path.exists():
        print(f"[WARN] YAML not found: {yaml_path}")
        continue

    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    train_rel = data_cfg.get("train")
    if train_rel is None:
        print(f"[WARN] 'train' key not found in YAML: {yaml_path}")
        continue

    train_dir = BASE_DIR / f"yolo_split_{fold_name}" / train_rel
    if not train_dir.exists():
        print(f"[WARN] Invalid or missing train path: {train_dir}")
        continue

    dataset = GPRDataset(train_dir)
    if len(dataset) == 0:
        print(f"[WARN] Empty dataset: {train_dir}")
        continue

    print(f"\n📂 Training AE on {fold_name}: {train_dir} (N={len(dataset)})")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        drop_last=False,
    )

    ae = AutoEncoder(loss_type="hybrid").to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)

    best_loss = float("inf")
    best_path = SAVE_DIR / f"pretrained_ae_{fold_name}.pth"

    for epoch in range(EPOCHS):
        ae.train()
        total_loss = 0.0
        total_psnr = 0.0
        count = 0

        for imgs in dataloader:
            imgs = imgs.to(device, non_blocking=True)  # (B,1,H,W) expected

            # 순전파 + 손실 + 재구성 함께 획득
            three_out, i_hat, loss = ae(imgs, return_loss=True, return_recon=True)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                psnr = compute_psnr(imgs, i_hat)  # (B,1,H,W) vs (B,1,H,W)
                total_psnr += psnr.item() * imgs.size(0)
                count += imgs.size(0)

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        avg_psnr = total_psnr / max(1, count)
        print(f"[Fold {fold_name}] Epoch {epoch + 1:02d} | AE Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

        if avg_loss < best_loss - 1e-6:  # best(Train 기준) 저장 — 가능하면 val 기준으로 확장
            best_loss = avg_loss
            torch.save(ae.state_dict(), best_path)

    # 마지막 가중치도 별도 저장
    final_path = SAVE_DIR / f"pretrained_ae_{fold_name}_last.pth"
    torch.save(ae.state_dict(), final_path)
    print(f"✅ Saved AE (final): {final_path.name}")
    if best_path.exists():
        print(f"✅ Saved AE (best):  {best_path.name}")
