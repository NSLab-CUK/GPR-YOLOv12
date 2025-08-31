# ===============================
# File: ultralytics/nn/modules/ae.py  (your AutoEncoder impl)
# ===============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.losses_ae import (
    hybrid_loss, ssim_loss, structural_loss, perceptual_loss
)


class AutoEncoder(nn.Module):
    def __init__(self, loss_type: str = "hybrid"):
        super().__init__()
        self.loss_type = loss_type.lower()

        # Encoder: 1 → 64 → 128 → 256
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
        )

        # Bottleneck: 256 → 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
        )

        # Decoder: 512 → 256 → 128 → 64 → 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.Sigmoid(),
        )

    # -----------------------------------
    # Utilities
    # -----------------------------------
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Tuple/dtype/scale/3ch→1ch 안전 전처리."""
        if isinstance(x, tuple):  # dataloader에서 (img, ..) 형태일 때 방어
            x = x[0]
        if x.dtype != torch.float32:
            x = x.float()
        # 0~255 스케일이면 0~1로 맞춤
        if torch.is_tensor(x) and x.max() > 1.5:
            x = x / 255.0
        # AE는 1ch 기준. 3ch가 들어오면 평균으로 1ch로 강제(의미적 RGB가 아님을 감안해 단순 평균)
        if x.ndim == 4 and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        # 최종 체크
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"AutoEncoder expects (B,1,H,W); got {tuple(x.shape)}")
        return x

    def minmax_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        min_val = x_flat.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        max_val = x_flat.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        return (x - min_val) / (max_val - min_val + eps)

    # -----------------------------------
    # Forward
    # -----------------------------------
    def forward(self, x: torch.Tensor, return_loss: bool = False, return_recon: bool = False):
        x = self._preprocess(x)

        i = x  # original input (B,1,H,W)
        enc = self.encoder(i)
        bottleneck = self.bottleneck(enc)
        i_hat = self.decoder(bottleneck)            # (B,1,H,W)

        # YOLO 통합용 3채널 출력: [i, i_hat_norm, |i-i_hat|_norm]
        diff_norm = self.minmax_norm(torch.abs(i - i_hat))
        i_hat_norm = self.minmax_norm(i_hat)
        three_channel_output = torch.cat([i, i_hat_norm, diff_norm], dim=1)  # (B,3,H,W)

        if return_loss:
            if self.loss_type == "l1":
                loss = F.l1_loss(i_hat, i)
            elif self.loss_type == "ssim":
                loss = ssim_loss(i_hat, i)
            elif self.loss_type == "structural":
                loss = structural_loss(i_hat, i)
            elif self.loss_type == "perceptual":
                loss = perceptual_loss(i_hat, i)
            elif self.loss_type == "hybrid":
                loss = hybrid_loss(i_hat, i)
            else:
                raise ValueError(f"Invalid loss_type: {self.loss_type}")

            if return_recon:
                return three_channel_output, i_hat, loss
            return three_channel_output, loss

        else:
            if return_recon:
                return three_channel_output, i_hat
            return three_channel_output

    def get_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """PSNR 등 계산용으로 순수 재구성(i_hat)만 반환."""
        x = self._preprocess(x)
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        i_hat = self.decoder(bottleneck)
        return i_hat