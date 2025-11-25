# prototype_analysis.py
"""
Prototype vs 疾病多标签的频率分析与可视化。

输入：
- ssl_outputs/features_meta_with_cluster.csv   # 每张图像的 patient_id + image_path + cluster_id
- ssl_outputs/prototypes_centers.npy          # Prototype 中心（可选）
- ../labels.xlsx                              # 患者级多标签（id + 4 个疾病）

输出：
- analysis_outputs/patient_prototype_hist.npy        # [N_patients, K]
- analysis_outputs/patient_labels.npy                # [N_patients, 4]
- analysis_outputs/patient_ids.npy                   # [N_patients]
- analysis_outputs/prototype_stats_<疾病名>.csv      # 每个疾病对应的 prototype 频率差
- 若干 PNG 可视化
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ================ 路径配置 ================

BASE_DIR = "."  # 当前文件所在目录：code/analysis_02/
SSL_OUTPUT_DIR = os.path.join(BASE_DIR, "ssl_outputs")
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_outputs")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

META_WITH_CLUSTER_CSV = os.path.join(SSL_OUTPUT_DIR, "features_meta_with_cluster.csv")
PROTOTYPE_CENTERS_NPY = os.path.join(SSL_OUTPUT_DIR, "prototypes_centers.npy")
LABELS_XLSX = "../labels.xlsx"   # 相对于 code/analysis_02/

PATIENT_ID_COL = "id"            # labels.xlsx 中患者 id 列名
DISEASE_COLS = ["乙型肝炎", "糖尿病", "血脂异常", "高尿酸血症"]   # 4 个多标签列名

# ===== 中文字体适配 =====
# 尝试优先使用常见的中文字体（Windows 下一般有 SimHei 或 Microsoft YaHei）
zh_fonts = ["SimHei", "Microsoft YaHei", "MSYH", "STSong"]
found_font = None
for f in zh_fonts:
    if f in [ft.name for ft in font_manager.fontManager.ttflist]:
        found_font = f
        break

if found_font is not None:
    plt.rcParams["font.sans-serif"] = [found_font]  # 用找到的中文字体
else:
    # 实在没找到就用默认字体，但此时中文可能仍显示不全
    logging.warning("未找到可用的中文字体，图像中的中文可能无法正常显示。")

# 解决负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False
# =======================


import logging
import os

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "prototype_analysis.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ================ Step 1: 构建患者级 prototype 直方图 ================

def build_patient_prototype_histograms():
    """
    从 meta_with_cluster + labels.xlsx 构建：
    - patient_ids: [N_patients]
    - H: [N_patients, K]  每个患者的 prototype 频率向量
    - Y: [N_patients, 4]  多标签矩阵
    并保存到 analysis_outputs/ 方便后续分析。
    """
    if not os.path.exists(META_WITH_CLUSTER_CSV):
        raise FileNotFoundError(f"未找到 {META_WITH_CLUSTER_CSV}，请先运行 prototype_ssl.py 的 cluster 步骤。")

    meta_df = pd.read_csv(META_WITH_CLUSTER_CSV)
    logging.info(f"[build_hist] 读取 meta_with_cluster：{meta_df.shape[0]} 条记录")

    if "cluster_id" not in meta_df.columns:
        raise ValueError("meta_with_cluster.csv 中缺少 'cluster_id' 列，请检查 prototype_ssl.py 的输出。")

    # 保证 patient_id 列存在（来自 features_meta_with_cluster）
    if "patient_id" not in meta_df.columns:
        raise ValueError("meta_with_cluster.csv 中缺少 'patient_id' 列。")

    # 读取 labels.xlsx
    labels_df = pd.read_excel(LABELS_XLSX)
    logging.info(f"[build_hist] 读取 labels.xlsx：{labels_df.shape[0]} 个患者")

    # 只保留在 labels 和 meta 中都出现过的患者
    # 两边的 id 先转成 int（如果可以的话）
    def to_int_if_possible(x):
        try:
            return int(x)
        except Exception:
            return x

    meta_df["patient_id_norm"] = meta_df["patient_id"].apply(to_int_if_possible)
    labels_df["patient_id_norm"] = labels_df[PATIENT_ID_COL].apply(to_int_if_possible)

    # 内连接，找交集
    patient_ids_in_both = sorted(set(meta_df["patient_id_norm"]).intersection(set(labels_df["patient_id_norm"])))
    logging.info(f"[build_hist] 共有 {len(patient_ids_in_both)} 个患者在图像和标签中都有记录。")

    # 映射 patient_id -> row index
    pid_to_idx = {pid: idx for idx, pid in enumerate(patient_ids_in_both)}
    num_patients = len(patient_ids_in_both)

    # 读取 prototype 簇数 K
    if not os.path.exists(PROTOTYPE_CENTERS_NPY):
        raise FileNotFoundError(f"未找到 {PROTOTYPE_CENTERS_NPY}，请先运行 prototype_ssl.py 的 cluster 步骤。")
    centers = np.load(PROTOTYPE_CENTERS_NPY)  # [K, feat_dim]
    K = centers.shape[0]
    logging.info(f"[build_hist] 检测到 prototype 数 K={K}")

    # 初始化直方图矩阵 H
    H_counts = np.zeros((num_patients, K), dtype=np.float32)

    # 遍历每张图，给对应患者的 cluster 计数
    for _, row in meta_df.iterrows():
        pid = row["patient_id_norm"]
        if pid not in pid_to_idx:
            continue
        idx = pid_to_idx[pid]
        cid = int(row["cluster_id"])
        if 0 <= cid < K:
            H_counts[idx, cid] += 1.0

    # 归一化成频率（可选）
    H = H_counts / (H_counts.sum(axis=1, keepdims=True) + 1e-12)

    # 构建标签矩阵 Y
    # 先构建 patient_id_norm -> 标签行
    labels_df = labels_df.set_index("patient_id_norm")
    Y_list = []
    for pid in patient_ids_in_both:
        # 有些患者可能某个疾病列缺失，统一处理为 0
        row = labels_df.loc[pid]
        labels = []
        for col in DISEASE_COLS:
            if col in row:
                val = row[col]
                try:
                    val = int(val)
                except Exception:
                    val = 0
            else:
                val = 0
            labels.append(val)
        Y_list.append(labels)

    Y = np.array(Y_list, dtype=np.int64)  # [N_patients, 4]

    # 保存结果
    H_path = os.path.join(ANALYSIS_OUTPUT_DIR, "patient_prototype_hist.npy")
    Y_path = os.path.join(ANALYSIS_OUTPUT_DIR, "patient_labels.npy")
    Pid_path = os.path.join(ANALYSIS_OUTPUT_DIR, "patient_ids.npy")

    np.save(H_path, H)
    np.save(Y_path, Y)
    np.save(Pid_path, np.array(patient_ids_in_both))

    logging.info(f"[build_hist] 患者 prototype 频率矩阵 H 保存到 {H_path}，形状 = {H.shape}")
    logging.info(f"[build_hist] 患者标签矩阵 Y 保存到 {Y_path}，形状 = {Y.shape}")
    logging.info(f"[build_hist] 患者 id 序列保存到 {Pid_path}")


# ================ Step 2: 分析 prototype 与疾病的关系 ================

def analyze_prototype_vs_disease(top_k: int = 15):
    """
    对每个疾病标签：
    - 分阳性组 vs 阴性组
    - 计算每个 prototype 在两组中的平均频率差
    - 输出 CSV 和简单条形图
    """
    H_path = os.path.join(ANALYSIS_OUTPUT_DIR, "patient_prototype_hist.npy")
    Y_path = os.path.join(ANALYSIS_OUTPUT_DIR, "patient_labels.npy")

    if not (os.path.exists(H_path) and os.path.exists(Y_path)):
        raise FileNotFoundError("请先运行 build_patient_prototype_histograms() 构建 H 和 Y。")

    H = np.load(H_path)  # [N_patients, K]
    Y = np.load(Y_path)  # [N_patients, 4]
    N, K = H.shape
    logging.info(f"[analyze] 加载 H: {H.shape}, Y: {Y.shape}")

    # 对每个疾病列做分析
    for d_idx, d_name in enumerate(DISEASE_COLS):
        y = Y[:, d_idx]   # [N_patients]
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        logging.info(f"[analyze] 疾病 {d_name}: 阳性 {num_pos} 例, 阴性 {num_neg} 例")

        if num_pos == 0 or num_neg == 0:
            logging.warning(f"[analyze] 疾病 {d_name} 阳性或阴性样本数为 0，跳过分析。")
            continue

        H_pos = H[pos_mask]  # [N_pos, K]
        H_neg = H[neg_mask]  # [N_neg, K]

        mean_pos = H_pos.mean(axis=0)  # [K]
        mean_neg = H_neg.mean(axis=0)  # [K]
        diff = mean_pos - mean_neg     # [K]

        # 整理成 DataFrame
        df_stats = pd.DataFrame({
            "prototype_id": np.arange(K),
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "diff_pos_minus_neg": diff
        })

        # 按差异绝对值排序，方便找“最相关 prototype”
        df_stats_sorted = df_stats.reindex(df_stats["diff_pos_minus_neg"].abs().sort_values(ascending=False).index)

        csv_path = os.path.join(ANALYSIS_OUTPUT_DIR, f"prototype_stats_{d_name}.csv")
        df_stats_sorted.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logging.info(f"[analyze] 疾病 {d_name} 的 prototype 统计保存到 {csv_path}")

        # 画一个 top_k prototype 的条形图（diff）
        top_df = df_stats_sorted.head(top_k).copy()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(top_df)), top_df["diff_pos_minus_neg"])
        plt.xticks(range(len(top_df)), top_df["prototype_id"], rotation=45)
        plt.xlabel("Prototype ID")
        plt.ylabel("mean_pos - mean_neg")
        plt.title(f"Top {top_k} Prototypes for {d_name} (freq diff)")
        plt.tight_layout()

        png_path = os.path.join(ANALYSIS_OUTPUT_DIR, f"prototype_diff_top{top_k}_{d_name}.png")
        plt.savefig(png_path, dpi=200)
        plt.close()
        logging.info(f"[analyze] 疾病 {d_name} 的 top-{top_k} 差异条形图保存到 {png_path}")


# ================ Step 3: 可选 - 总体热力图 ================

def plot_global_heatmap():
    """
    画一个 prototype × 疾病 的热力图：
    - 每个 cell = diff_pos_minus_neg（或 mean_pos）
    """
    H_path = os.path.join(ANALYSIS_OUTPUT_DIR, "patient_prototype_hist.npy")
    Y_path = os.path.join(ANALYSIS_OUTPUT_DIR, "patient_labels.npy")

    if not (os.path.exists(H_path) and os.path.exists(Y_path)):
        logging.warning("未找到 H 或 Y，跳过全局热力图绘制。")
        return

    H = np.load(H_path)
    Y = np.load(Y_path)
    N, K = H.shape

    diff_matrix = np.zeros((len(DISEASE_COLS), K), dtype=np.float32)

    for d_idx, d_name in enumerate(DISEASE_COLS):
        y = Y[:, d_idx]
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            logging.warning(f"[heatmap] 疾病 {d_name} 阳性或阴性样本数为 0，置 diff 为 0。")
            continue

        H_pos = H[pos_mask]
        H_neg = H[neg_mask]

        mean_pos = H_pos.mean(axis=0)
        mean_neg = H_neg.mean(axis=0)
        diff = mean_pos - mean_neg

        diff_matrix[d_idx] = diff

    plt.figure(figsize=(max(8, K / 4), 4 + len(DISEASE_COLS)))
    im = plt.imshow(diff_matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.02, pad=0.02)

    plt.yticks(range(len(DISEASE_COLS)), DISEASE_COLS)
    plt.xticks(range(K), range(K), rotation=90)
    plt.xlabel("Prototype ID")
    plt.title("Prototype frequency difference (mean_pos - mean_neg) heatmap")

    plt.tight_layout()
    png_path = os.path.join(ANALYSIS_OUTPUT_DIR, "prototype_diff_heatmap.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    logging.info(f"[heatmap] 全局热力图保存到 {png_path}")


# ================ 主入口 ================

if __name__ == "__main__":
    logging.info("===== Prototype vs Disease 频率分析开始 =====")
    build_patient_prototype_histograms()
    analyze_prototype_vs_disease(top_k=15)
    plot_global_heatmap()
    logging.info("===== Prototype vs Disease 频率分析结束 =====")
