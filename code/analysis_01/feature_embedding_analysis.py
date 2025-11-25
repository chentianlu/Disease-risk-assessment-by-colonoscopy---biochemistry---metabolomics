import os
import glob
import logging

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ================== 基本配置 ==================
DATA_ROOT = "../../data"               # 患者图像根目录：../data/001, ../data/002, ...
LABELS_XLSX = "../labels.xlsx"       # 患者标签文件
WEIGHTS_PATH = "../../weight/efficientnet_b0_rwightman-3dd342df.pth"
OUTPUT_DIR = "."           # 所有结果放在 code/analysis/ 下

PATIENT_ID_COL = "id"               # labels.xlsx 里的患者 ID 列名（如果不是 id，就改这里）

# 4 个疾病配置：列名要和 labels.xlsx 完全一致
DISEASE_CONFIGS = [
    {"col": "乙型肝炎", "name_en": "hepatitisB"},
    {"col": "糖尿病", "name_en": "diabetes"},
    {"col": "血脂异常", "name_en": "dyslipidemia"},
    {"col": "高尿酸血症", "name_en": "hyperuricemia"},
]


# ================== 工具函数 ==================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )


# ================== 数据集 & 特征提取 ==================
class ImageListDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, path


def build_encoder(device):
    """
    构建 EfficientNet-B0 encoder（只做特征提取，输出 1280 维）
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=0,       # 去掉分类头，只保留特征
        global_pool="avg"
    )
    if os.path.isfile(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info(
            f"Loaded encoder weights. Missing keys: {len(missing)}, "
            f"Unexpected keys: {len(unexpected)}"
        )
    else:
        logging.warning(f"Weight file not found: {WEIGHTS_PATH}, using random init.")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def collect_patient_image_paths(labels_df):
    """
    根据 labels.xlsx 收集所有存在的患者图像路径
    返回：
        all_image_paths: List[str]
        image_patient_ids: List[str]  # 每张图对应的患者id（字符串格式）
    """
    all_image_paths = []
    image_patient_ids = []

    for _, row in labels_df.iterrows():
        pid = row[PATIENT_ID_COL]
        pid_str = str(pid).strip()
        # 统一使用 3 位补零的文件夹，如 001, 002, ...
        folder = os.path.join(DATA_ROOT, pid_str)

        if not os.path.isdir(folder):
            logging.warning(f"patient folder not found: {folder}")
            continue

        # 匹配常见图像后缀
        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            img_files.extend(glob.glob(os.path.join(folder, ext)))

        if len(img_files) == 0:
            logging.warning(f"no images found in folder: {folder}")
            continue

        all_image_paths.extend(img_files)
        image_patient_ids.extend([pid_str] * len(img_files))

    logging.info(f"Total images collected: {len(all_image_paths)}")
    return all_image_paths, image_patient_ids


def extract_features(encoder, image_paths, device, batch_size=32):
    """
    对所有图像跑一遍 EfficientNet-B0，得到 [N, 1280] 特征
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ImageListDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            f = encoder(imgs)          # [B, 1280]
            f = f.cpu().numpy()
            feats.append(f)

    feats = np.concatenate(feats, axis=0)
    logging.info(f"Feature shape: {feats.shape}")
    return feats


# ================== 降维  ==================
def run_dim_reduction(features, pca_dim=50, tsne_dim=2):
    """
    自动兼容 sklearn 新旧版本的 TSNE：
    - 旧版本需要 n_iter
    - 新版本会报错，因此跳过 n_iter
    """
    logging.info("Running PCA...")
    pca = PCA(n_components=pca_dim, random_state=42)
    feat_pca = pca.fit_transform(features)
    logging.info(f"PCA output shape: {feat_pca.shape}")

    logging.info("Running t-SNE (this may take several minutes)...")

    # 先尝试使用新 API（无 n_iter）
    try:
        tsne = TSNE(
            n_components=tsne_dim,
            perplexity=30,
            learning_rate=200,
            init="random",
            random_state=42,
            verbose=1,
        )
        feat_tsne = tsne.fit_transform(feat_pca)
        logging.info("[INFO] TSNE running (new API, without n_iter).")
    except TypeError:
        logging.info("[WARN] New TSNE API not supported, falling back to old TSNE API with n_iter.")

        # 旧版 sklearn TSNE API
        tsne = TSNE(
            n_components=tsne_dim,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            init="random",
            random_state=42,
            verbose=1,
        )
        feat_tsne = tsne.fit_transform(feat_pca)
        logging.info("[INFO] TSNE running (old API, with n_iter).")

    logging.info(f"TSNE output shape: {feat_tsne.shape}")
    return feat_pca, feat_tsne


def run_kmeans(feat_pca, n_clusters=3):
    logging.info("Running KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(feat_pca)
    unique, counts = np.unique(cluster_ids, return_counts=True)
    logging.info(f"Cluster counts: {dict(zip(unique, counts))}")
    return cluster_ids


# ================== 可视化 & 保存 ==================
def plot_tsne_for_disease(
    feat_tsne,
    labels_binary,
    cluster_ids,
    disease_name_en,
    save_path_png,
):
    """
    t-SNE 空间中：
      - 颜色表示疾病标签（阴性/阳性）
      - 点的形状表示 KMeans cluster
    """
    plt.figure(figsize=(8, 8))

    clusters = np.unique(cluster_ids)
    # 给每个 cluster 一个不同的 marker（不够用会循环）
    markers = ['o', 's', '^', 'D', 'P', 'X', '*']

    for i, cid in enumerate(clusters):
        marker = markers[i % len(markers)]
        mask_cluster = (cluster_ids == cid)

        # 这个 cluster 里的阴性样本
        mask_neg = (labels_binary == 0) & mask_cluster
        # 这个 cluster 里的阳性样本
        mask_pos = (labels_binary == 1) & mask_cluster

        # 阴性：蓝色
        if mask_neg.sum() > 0:
            plt.scatter(
                feat_tsne[mask_neg, 0],
                feat_tsne[mask_neg, 1],
                s=8,
                alpha=0.4,
                marker=marker,
                label=f"cluster {cid} - negative",
            )

        # 阳性：红色
        if mask_pos.sum() > 0:
            plt.scatter(
                feat_tsne[mask_pos, 0],
                feat_tsne[mask_pos, 1],
                s=10,
                alpha=0.8,
                marker=marker,
                label=f"cluster {cid} - positive",
            )

    plt.title(f"t-SNE colored by {disease_name_en} & KMeans clusters")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # 去重 legend（因为同一个 label 可能重复）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path_png, dpi=300)
    plt.close()
    logging.info(f"Embedding plot (with clusters) saved to {save_path_png}")



def save_embedding_table(
    image_paths,
    patient_ids,
    feat_tsne,
    cluster_ids,
    labels_binary,
    disease_name_en,
    save_path_csv,
):
    """
    保存一个 CSV：
        image_path, patient_id, f0..f1279, tsne_x, tsne_y, cluster_id, label_<disease>
    """
    
    data = {
        "image_path": image_paths,
        "patient_id": patient_ids,
        "tsne_x": feat_tsne[:, 0],
        "tsne_y": feat_tsne[:, 1],
        "cluster_id": cluster_ids,
        f"label_{disease_name_en}": labels_binary,
    }

    

    df = pd.DataFrame(data)
    df.to_csv(save_path_csv, index=False, encoding="utf-8-sig")
    logging.info(f"Saved embedding table to {save_path_csv}")


# ================== 主流程 ==================
def main():
    setup_logger()
    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 1. 读取患者标签
    labels_df = pd.read_excel(LABELS_XLSX)
    logging.info(f"Loaded labels: {len(labels_df)} patients")

    # 建一个 patient_id -> 该行记录 的字典，后面根据 patient_id 给每张图赋标签
    labels_df[PATIENT_ID_COL] = labels_df[PATIENT_ID_COL].astype(str).str.strip()
    patient_row_dict = {
        row[PATIENT_ID_COL]: row for _, row in labels_df.iterrows()
    }

    # 2. 收集所有图像路径
    all_image_paths, image_patient_ids = collect_patient_image_paths(labels_df)
    if len(all_image_paths) == 0:
        logging.error("No images found, abort.")
        return

    # 3. 构建并加载 encoder
    encoder = build_encoder(device)

    # 4. 对所有图像跑一次特征
    features = extract_features(encoder, all_image_paths, device, batch_size=32)

    # 5. 在整套特征上做一次 PCA + t-SNE + KMeans
    feat_pca, feat_tsne = run_dim_reduction(features, pca_dim=50, tsne_dim=2)
    cluster_ids = run_kmeans(feat_pca, n_clusters=3)

    # 6. 对每个疾病，单独做可视化与结果保存
    for cfg in DISEASE_CONFIGS:
        col = cfg["col"]
        name_en = cfg["name_en"]

        # 为每张图生成一个 0/1 标签（根据其 patient_id 在 labels_df 中对应的该疾病列）
        labels_binary = []
        for pid in image_patient_ids:
            row = patient_row_dict.get(pid, None)
            if row is None or pd.isna(row[col]):
                # 缺失标签的，默认当阴性（也可以选择跳过）
                labels_binary.append(0)
            else:
                # 假设 labels.xlsx 中该列是 0/1
                v = int(row[col])
                labels_binary.append(1 if v == 1 else 0)
        labels_binary = np.array(labels_binary, dtype=np.int64)
        logging.info(
            f"[{name_en}] positives: {labels_binary.sum()}, "
            f"negatives: {len(labels_binary) - labels_binary.sum()}"
        )

        # 画 t-SNE 图
        png_path = os.path.join(OUTPUT_DIR, f"embedding_{name_en}_tsne.png")
        plot_tsne_for_disease(
            feat_tsne=feat_tsne,
            labels_binary=labels_binary,
            cluster_ids=cluster_ids,
            disease_name_en=name_en,
            save_path_png=png_path,
        )

        # 保存 CSV
        csv_path = os.path.join(OUTPUT_DIR, f"embedding_{name_en}_features.csv")
        save_embedding_table(
            image_paths=all_image_paths,
            patient_ids=image_patient_ids,
            # features=features,
            feat_tsne=feat_tsne,
            cluster_ids=cluster_ids,
            labels_binary=labels_binary,
            disease_name_en=name_en,
            save_path_csv=csv_path,
        )


if __name__ == "__main__":
    main()
