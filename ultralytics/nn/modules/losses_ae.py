# losses_ae.py
import torch
import torch.nn.functional as F
from torchvision import models
import math  # 함수 안에서 사용 가능 (또는 파일 맨 위)

# ─── Global VGG16 Feature Extractor (Perceptual Loss용) ───
vgg = models.vgg16(pretrained=True).features[:16]
vgg.eval()
for p in vgg.parameters():
    p.requires_grad = False

# ─── Structural Loss (L1) ───
def structural_loss(x_hat, x):
    return F.l1_loss(x_hat, x)

# ─── SSIM Loss ───
def ssim_loss(x, y, window_size=11):

    def gaussian(size, sigma):
        gauss = torch.tensor([math.exp(-(x - size // 2)**2 / (2 * sigma**2)) for x in range(size)])
        return gauss / gauss.sum()

    def create_window(size, channel):
        _1d = gaussian(size, 1.5).unsqueeze(1)
        _2d = _1d @ _1d.T
        window = _2d.expand(channel, 1, size, size).contiguous()
        return window


    channel = x.shape[1]
    window = create_window(window_size, channel).to(x.device)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channel)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu_xy

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = ssim_n / ssim_d
    return 1 - ssim.mean()

# ─── Perceptual Loss ───
def perceptual_loss(x_hat, x):
    x_hat_up = F.interpolate(x_hat.repeat(1, 3, 1, 1), size=(224, 224), mode='bilinear')
    x_up = F.interpolate(x.repeat(1, 3, 1, 1), size=(224, 224), mode='bilinear')

    # ⭐ VGG를 입력과 같은 디바이스로 이동
    vgg_device = x.device
    vgg.to(vgg_device)

    return F.l1_loss(vgg(x_hat_up.to(vgg_device)), vgg(x_up.to(vgg_device)))

# ─── Hybrid Loss ───
def hybrid_loss(x_hat, x, alpha=0.5, beta=0.3, gamma=0.2):
    return (
        alpha * structural_loss(x_hat, x) +
        beta * ssim_loss(x_hat, x) +
        gamma * perceptual_loss(x_hat, x)
    )




