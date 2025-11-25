# prototype_ssl.py
"""
自监督 Prototype Discovery：
1）SimCLR 自监督预训练（可选）
2）全体图像特征抽取（高速 GPU 版本）
3）KMeans 聚类得到 prototype（视觉原型）

相对路径说明：本文件位于 code/analysis_02/
DATA_ROOT = "../../data"       # 患者图像根目录：.../data/001, 002, ...
WEIGHTS_PATH = "../../weight/efficientnet_b0_rwightman-3dd342df.pth"
"""

import os
import glob
import argparse
import logging
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.cluster import KMeans

# ================== 基本配置 ==================

DATA_ROOT = "../../data"  # 患者图像根目录
WEIGHTS_PATH = "../../weight/efficientnet_b0_rwightman-3dd342df.pth"

OUTPUT_DIR = "./ssl_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# SimCLR 训练超参数
IMG_SIZE = 224
BATCH_SIZE = 32           # 只用于 pretrain
EXTRACT_BATCH_SIZE = 128  # ✅ 特征抽取 batch（可以按显存调大/调小）
EPOCHS = 50
LR = 1e-3
TEMPERATURE = 0.5
PROJ_DIM = 128

NUM_CLUSTERS = 128

# ================== 日志 ==================

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "prototype_ssl.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# cuDNN 自动优化卷积算法
torch.backends.cudnn.benchmark = True


# ================== Dataset 定义 ==================

class SimCLRDataset(Dataset):
    """
    用于 SimCLR 自监督训练的数据集：
    - 遍历 DATA_ROOT 下所有图像（不需要标签）
    - 每次返回同一张图的两种不同增强 view
    """
    def __init__(self, data_root: str, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = self._collect_image_paths()

        logging.info(f"[SimCLRDataset] 共找到 {len(self.image_paths)} 张图像用于自监督训练。")

    def _collect_image_paths(self) -> List[str]:
        paths = []
        for pid_dir in sorted(os.listdir(self.data_root)):
            full_dir = os.path.join(self.data_root, pid_dir)
            if not os.path.isdir(full_dir):
                continue
            for ext in IMG_EXTS:
                paths.extend(glob.glob(os.path.join(full_dir, f"*{ext}")))
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is None:
            raise ValueError("SimCLRDataset 需要提供 transform")

        # 同一张原图，两次独立增强
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


class AllImagesDataset(Dataset):
    """
    用于特征抽取的 Dataset：
    - 每个样本：图像张量 + patient_id + image_path
    """
    def __init__(self, data_root: str, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = self._collect_samples()

        logging.info(f"[AllImagesDataset] 共找到 {len(self.samples)} 张图像用于特征抽取。")

    def _collect_samples(self):
        samples = []
        for pid_dir in sorted(os.listdir(self.data_root)):
            full_dir = os.path.join(self.data_root, pid_dir)
            if not os.path.isdir(full_dir):
                continue
            try:
                patient_id = int(pid_dir)
            except ValueError:
                patient_id = pid_dir

            for ext in IMG_EXTS:
                for img_path in glob.glob(os.path.join(full_dir, f"*{ext}")):
                    samples.append((patient_id, img_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is None:
            raise ValueError("AllImagesDataset 需要提供 transform")

        x = self.transform(img)
        return x, patient_id, img_path


# ================== 图像增强 ==================

def get_simclr_transform(img_size: int = 224):
    color_jitter = transforms.ColorJitter(
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2,
    )
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return data_transforms


def get_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ================== SimCLR 模型定义 ==================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, weights_path: str = None, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b0",
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.encoder.num_features

        if weights_path is not None and os.path.exists(weights_path):
            logging.info(f"[SimCLRModel] 从 {weights_path} 加载预训练权重（strict=False）")
            state_dict = torch.load(weights_path, map_location="cpu")
            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
            logging.info(f"[SimCLRModel] Loaded encoder weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            logging.info("[SimCLRModel] 未提供预训练权重，将从随机初始化开始自监督。")

        self.projection_head = ProjectionHead(in_dim=feat_dim, proj_dim=proj_dim)

    def forward(self, x):
        feat = self.encoder(x)
        z = self.projection_head(feat)
        z = F.normalize(z, dim=1)
        return z, feat


# ================== NT-Xent Loss ==================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)   # [2B, D]

    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    pos_indices = torch.arange(batch_size, device=z.device)
    pos_indices = torch.cat([pos_indices + batch_size, pos_indices], dim=0)

    labels = pos_indices
    loss = F.cross_entropy(sim, labels)
    return loss


# ================== 1. 自监督预训练 ==================

def run_pretrain(device: torch.device):
    logging.info("===== Step 1: SimCLR 自监督预训练开始 =====")

    transform = get_simclr_transform(IMG_SIZE)
    dataset = SimCLRDataset(DATA_ROOT, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = SimCLRModel(weights_path=WEIGHTS_PATH, proj_dim=PROJ_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for step, (x1, x2) in enumerate(dataloader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            z1, _ = model(x1)
            z2, _ = model(x2)
            loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 50 == 0:
                logging.info(f"[Epoch {epoch}/{EPOCHS}] step {step}, loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"=== Epoch {epoch}/{EPOCHS} completed, avg_loss={avg_loss:.4f} ===")

        ckpt_path = os.path.join(OUTPUT_DIR, f"simclr_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"[Checkpoint] Saved to {ckpt_path}")

    logging.info("===== Step 1: SimCLR 自监督预训练结束 =====")


# ================== 2. 特征抽取（高速 GPU 版本） ==================

def run_extract(device: torch.device, ckpt_path: str = None):
    logging.info("===== Step 2: 特征抽取开始 =====")

    # 1）构建 encoder
    encoder = timm.create_model(
        "efficientnet_b0",
        num_classes=0,
        global_pool="avg"
    )

    # 先加载 ImageNet 预训练权重
    if WEIGHTS_PATH is not None and os.path.exists(WEIGHTS_PATH):
        logging.info(f"[Feature Extract] 从 {WEIGHTS_PATH} 加载 encoder 预训练权重（strict=False）")
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        logging.info(f"[Feature Extract] Loaded encoder weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # 再加载自监督 ckpt（只覆盖 encoder 部分）
    if ckpt_path is not None and os.path.exists(ckpt_path):
        logging.info(f"[Feature Extract] 从自监督 checkpoint {ckpt_path} 加载权重（只取 encoder 部分）")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder_state = {k.replace("encoder.", ""): v
                         for k, v in ckpt.items()
                         if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        logging.info(f"[Feature Extract] Loaded encoder from SimCLR. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    encoder = encoder.to(device)
    encoder.eval()

    # 2）DataLoader
    transform = get_eval_transform(IMG_SIZE)
    dataset = AllImagesDataset(DATA_ROOT, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=EXTRACT_BATCH_SIZE,
        shuffle=False,
        num_workers=4,          # 可根据机器改成 4 / 8
        pin_memory=True
    )

    all_feats = []
    meta_records = []

    # ✅ 使用 inference_mode，速度略快，显存更省
    with torch.inference_mode():
        for step, (x, patient_ids, img_paths) in enumerate(dataloader, start=1):
            x = x.to(device, non_blocking=True)

            feats = encoder(x)              # [B, feat_dim]
            feats = feats.cpu().numpy()
            all_feats.append(feats)

            # 修复 patient_id 写成 tensor(xx) 的问题
            for pid, path in zip(patient_ids, img_paths):
                # DataLoader 会把 patient_id 变成 tensor/int，这里统一处理
                if isinstance(pid, torch.Tensor):
                    pid = pid.item()
                # 再安全转 int 或 str
                try:
                    pid_clean = int(pid)
                except Exception:
                    pid_clean = str(pid)

                meta_records.append({
                    "patient_id": pid_clean,
                    "image_path": path
                })

            if step % 20 == 0:
                logging.info(f"[Feature Extract] step {step}/{len(dataloader)}")

    all_feats = np.concatenate(all_feats, axis=0)
    meta_df = pd.DataFrame(meta_records)

    feat_path = os.path.join(OUTPUT_DIR, "features_all.npy")
    meta_path = os.path.join(OUTPUT_DIR, "features_meta.csv")

    np.save(feat_path, all_feats)
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    logging.info(f"[Feature Extract] 特征保存到 {feat_path}，shape = {all_feats.shape}")
    logging.info(f"[Feature Extract] 元数据保存到 {meta_path}（含 patient_id, image_path）")
    logging.info("===== Step 2: 特征抽取结束 =====")


# ================== 3. KMeans 聚类 ==================

def run_cluster(num_clusters: int = NUM_CLUSTERS):
    logging.info("===== Step 3: KMeans 聚类（Prototype 学习）开始 =====")

    feat_path = os.path.join(OUTPUT_DIR, "features_all.npy")
    meta_path = os.path.join(OUTPUT_DIR, "features_meta.csv")

    if not os.path.exists(feat_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("请先运行特征抽取（mode=extract），确保 features_all.npy 和 features_meta.csv 存在。")

    feats = np.load(feat_path)
    meta_df = pd.read_csv(meta_path)

    logging.info(f"[Cluster] 加载特征：{feats.shape[0]} 个实例，特征维度 {feats.shape[1]}")

    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats_norm = feats / norms

    logging.info(f"[Cluster] 开始 KMeans 聚类，K={num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(feats_norm)

    centers = kmeans.cluster_centers_

    meta_df["cluster_id"] = cluster_ids

    meta_with_cluster_path = os.path.join(OUTPUT_DIR, "features_meta_with_cluster.csv")
    centers_path = os.path.join(OUTPUT_DIR, "prototypes_centers.npy")
    assign_path = os.path.join(OUTPUT_DIR, "prototypes_assignments.npy")

    meta_df.to_csv(meta_with_cluster_path, index=False, encoding="utf-8-sig")
    np.save(centers_path, centers)
    np.save(assign_path, cluster_ids)

    logging.info(f"[Cluster] meta+cluster 保存到 {meta_with_cluster_path}")
    logging.info(f"[Cluster] prototype centers 保存到 {centers_path}")
    logging.info(f"[Cluster] assignment 保存到 {assign_path}")
    logging.info("===== Step 3: KMeans 聚类结束 =====")


# ================== 主入口 ==================

def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised Prototype Discovery (SimCLR + KMeans)")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["pretrain", "extract", "cluster", "all"],
        help="运行模式：pretrain / extract / cluster / all"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="特征抽取时使用的自监督 checkpoint 路径（.pth）。如果为空则只用 ImageNet 预训练权重。"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=NUM_CLUSTERS,
        help="KMeans 聚类的簇数 K"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.mode in ["pretrain", "all"]:
        run_pretrain(device)

    if args.mode in ["extract", "all"]:
        run_extract(device, ckpt_path=args.ckpt)

    if args.mode in ["cluster", "all"]:
        run_cluster(num_clusters=args.num_clusters)
