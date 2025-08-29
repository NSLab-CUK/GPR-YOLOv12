# ae/gpr_dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class GPRDataset(Dataset):
    def __init__(self, image_dir, image_size=512):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # 입력이 이미 1채널이라면 (1, H, W) 유지됨
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(path)  # convert("L") 제거
        return self.transform(image)
