import os
import glob
import random
from typing import List

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

# 避免“图片被截断”报错
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PatientBagDataset(Dataset):
    """
    读取患者级“图像袋”（bag）。
    - root_dir: data 根目录，子目录为 patient_id（此处用 id）
    - labels_csv: 含 id 与多标签列（0/1），可为 .xlsx/.xls/.csv
    - max_images: 每位患者最多取多少张图（不足则重复或采样）
    - train: 训练模式会做更强的数据增强
    """
    def __init__(self, root_dir: str, labels_csv: str, label_cols: List[str],
                 max_images: int = 32, train: bool = True,
                 img_exts=(".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".JPG", ".PNG", ".JPEG", ".BMP", ".TIF", ".TIFF")):
        super().__init__()
        self.root_dir = root_dir
        self.df = self._read_labels(labels_csv)
        self.label_cols = label_cols
        self.max_images = max_images
        self.train = train
        self.img_exts = img_exts

        # ✅ 用 id 列作为患者标识（与你的表一致）
        if 'id' not in self.df.columns:
            raise KeyError("标签表缺少 'id' 列，请确认 labels 文件的列名。")
        self.patient_ids = self.df['id'].astype(str).tolist()

        # 多标签矩阵
        for c in self.label_cols:
            if c not in self.df.columns:
                raise KeyError(f"标签列 {c} 不在标签表中。现有列：{list(self.df.columns)}")
        self.labels = self.df[self.label_cols].values.astype('float32')

        # 变换：基础 + 轻量增强
        size = 256
        if train:
            self.tx = T.Compose([
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tx = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    @staticmethod
    def _read_labels(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        # 有些“csv”其实是excel导出的带中文/其他编码，这里兜底尝试
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_excel(path)

    def _load_images_for_patient(self, pid: str) -> List[str]:
        pdir = os.path.join(self.root_dir, pid)
        files = []
        if os.path.isdir(pdir):
            for ext in self.img_exts:
                files.extend(glob.glob(os.path.join(pdir, f"*{ext}")))
        files = sorted(files)
        return files

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        y = torch.from_numpy(self.labels[idx])  # [n_labels]
        paths = self._load_images_for_patient(pid)
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found for patient id={pid} in {os.path.join(self.root_dir, pid)}")

        # 采样/重复到固定长度
        if len(paths) >= self.max_images:
            paths_sampled = random.sample(paths, self.max_images)
        else:
            reps = (self.max_images + len(paths) - 1) // len(paths)
            paths_sampled = (paths * reps)[:self.max_images]

        imgs, orig_paths = [], []
        for p in paths_sampled:
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                # 遇到损坏图片：跳过，换一张补齐
                alt = random.choice(paths)
                img = Image.open(alt).convert('RGB')
                p = alt
            imgs.append(self.tx(img))
            orig_paths.append(p)

        bag = torch.stack(imgs, dim=0)  # [N, 3, H, W]
        return bag, y, pid, orig_paths
