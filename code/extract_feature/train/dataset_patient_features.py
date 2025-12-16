# dataset_patient_features.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


class PatientBagFeatureDataset(Dataset):
    """
    直接读取“实例特征表”来构造患者级 bag。
    每个样本（患者）返回：
      feats: [N, D]   (该患者 N 张图/tiles 的特征)
      y:     [L]      多标签
      pid:   str
      paths: List[str]  原始 image_path（用于解释/排查）

    特征表格式要求：
      - 必须有 patient_id 列
      - 必须有 image_path 列（可选但建议保留）
      - 必须有 feat_0...feat_{D-1} 列（或你自定义前缀 feat_）
    标签表格式要求：
      - 必须有 id 列（与你现有 PatientBagDataset 一致）:contentReference[oaicite:1]{index=1}
      - 多标签列名由 label_cols 指定
    """

    def __init__(
        self,
        features_table: str,
        labels_csv: str,
        label_cols,
        max_instances: int = 0,   # <=0 用全部；>0 截断到最多这么多实例（保持你 train.py 习惯）
        train: bool = True,       # train=True 时可做随机采样（与 dataset_patient.py 一致）
        feat_prefix: str = "feat_",
        patient_id_col: str = "patient_id",
        image_path_col: str = "image_path",
    ):
        super().__init__()
        self.label_cols = list(label_cols)
        self.max_instances = max_instances
        self.train = train
        self.feat_prefix = feat_prefix
        self.patient_id_col = patient_id_col
        self.image_path_col = image_path_col

        # 读特征表
        df_feat = _read_table(features_table)
        if self.patient_id_col not in df_feat.columns:
            raise ValueError(f"特征表缺少列: {self.patient_id_col}")
        if self.image_path_col not in df_feat.columns:
            # 允许没有 image_path，但解释时就没法回溯
            df_feat[self.image_path_col] = ""

        feat_cols = [c for c in df_feat.columns if c.startswith(self.feat_prefix)]
        if len(feat_cols) == 0:
            raise ValueError(f"特征表中找不到任何 {self.feat_prefix}* 列。")

        # 确保 feat_0..feat_k 的顺序是数值顺序
        def _feat_key(x: str):
            # x like feat_123
            try:
                return int(x.split(self.feat_prefix, 1)[1])
            except Exception:
                return 10**12

        feat_cols = sorted(feat_cols, key=_feat_key)
        self.feat_cols = feat_cols

        # 读标签表（与你 train.py / dataset_patient.py 逻辑一致）:contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
        df_lab = _read_table(labels_csv)
        if "id" not in df_lab.columns:
            raise ValueError("标签文件中必须包含一列名为 'id' 的患者编号。")
        for c in self.label_cols:
            if c not in df_lab.columns:
                raise ValueError(f"标签列 '{c}' 不在标签文件中。")

        df_lab["id"] = df_lab["id"].astype(str)
        self.df_lab = df_lab.set_index("id")

        # 只保留“特征表中出现过且标签表里也有”的患者
        df_feat[self.patient_id_col] = df_feat[self.patient_id_col].astype(str)
        pids_feat = df_feat[self.patient_id_col].unique().tolist()
        pids = [pid for pid in pids_feat if pid in self.df_lab.index]
        if len(pids) == 0:
            raise RuntimeError("特征表的 patient_id 与标签表的 id 没有交集。")

        self.patient_ids = sorted(pids, key=lambda x: int(x) if x.isdigit() else x)

        # 为了快速按 pid 取子表：预建 group 索引
        self.groups = {pid: g for pid, g in df_feat.groupby(self.patient_id_col, sort=False)}

        # 记录特征维度
        self.feature_dim = len(self.feat_cols)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        g = self.groups.get(pid, None)
        if g is None or len(g) == 0:
            raise RuntimeError(f"患者 {pid} 在特征表里没有任何实例行。")

        # 取该患者的所有实例
        feats = g[self.feat_cols].values.astype("float32")  # [N, D]
        paths = g[self.image_path_col].astype(str).tolist()

        # max_instances 逻辑：训练随机采样，验证取前 K（与你 dataset_patient.py 一致）:contentReference[oaicite:4]{index=4}
        if self.max_instances is not None and self.max_instances > 0 and feats.shape[0] > self.max_instances:
            if self.train:
                sel = np.random.choice(feats.shape[0], self.max_instances, replace=False)
                feats = feats[sel]
                paths = [paths[i] for i in sel]
            else:
                feats = feats[: self.max_instances]
                paths = paths[: self.max_instances]

        # 标签
        y = self.df_lab.loc[pid, self.label_cols].values.astype("float32")  # [L]

        feats_t = torch.from_numpy(feats)   # [N, D]
        y_t = torch.from_numpy(y)           # [L]
        return feats_t, y_t, pid, paths
