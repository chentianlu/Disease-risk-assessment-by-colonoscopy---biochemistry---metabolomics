# train_ce_mil_from_xlsx.py
# 单文件：从 xlsx/csv 特征表训练 MIL（二分类：softmax prob1 + CrossEntropyLoss）
import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------ Reproducibility ------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # csv: 多编码兜底
    for enc in ["utf-8", "gbk", "ansi", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法读取: {path}")


# ------------------ Dataset (bag by patient) ------------------
class PatientBagFeatureDataset(Dataset):
    """
    从特征表(features_xlsx)按 patient_id 组织成患者 bag。
    每个 item：features [N,D], label int(0/1), patient_id str
    """
    def __init__(
        self,
        features_table: str,
        labels_table: str,
        label_col: str,
        patient_id_col_feat: str = "patient_id",
        patient_id_col_label: str = "id",
        feat_prefix: str = "feat_",
        max_instances: int = 0,
        train: bool = True,
    ):
        self.features_table = features_table
        self.labels_table = labels_table
        self.label_col = label_col
        self.pid_feat = patient_id_col_feat
        self.pid_lab = patient_id_col_label
        self.feat_prefix = feat_prefix
        self.max_instances = int(max_instances)
        self.train = train

        df_feat = load_table(features_table)
        df_lab = load_table(labels_table)

        if self.pid_feat not in df_feat.columns:
            raise ValueError(f"特征表缺少列 '{self.pid_feat}'")
        if self.pid_lab not in df_lab.columns:
            raise ValueError(f"标签表缺少列 '{self.pid_lab}'")
        if label_col not in df_lab.columns:
            raise ValueError(f"标签表缺少标签列 '{label_col}'")

        # 规范化 id
        df_feat[self.pid_feat] = df_feat[self.pid_feat].astype(str)
        df_lab[self.pid_lab] = df_lab[self.pid_lab].astype(str)

        # 取 feat 列并排序
        feat_cols = [c for c in df_feat.columns if c.startswith(self.feat_prefix)]
        if not feat_cols:
            raise ValueError(f"特征表中找不到以 '{self.feat_prefix}' 开头的特征列")
        def feat_key(c):
            s = c[len(self.feat_prefix):]
            return int(s) if s.isdigit() else s
        feat_cols = sorted(feat_cols, key=feat_key)
        self.feat_cols = feat_cols
        self.feature_dim = len(feat_cols)

        # 标签：转 0/1 int
        df_lab[label_col] = df_lab[label_col].fillna(0)
        try:
            df_lab[label_col] = df_lab[label_col].astype(int)
        except Exception:
            # 兜底：字符串转
            def to01(x):
                xs = str(x).strip().lower()
                if xs in ["1", "yes", "true", "是", "有", "positive", "患病"]:
                    return 1
                return 0
            df_lab[label_col] = df_lab[label_col].apply(to01).astype(int)

        # 对齐：只保留两边都有的患者
        feat_pids = set(df_feat[self.pid_feat].unique().tolist())
        df_lab = df_lab[df_lab[self.pid_lab].isin(feat_pids)].copy()
        if len(df_lab) == 0:
            raise RuntimeError("标签表与特征表没有任何患者ID交集，请检查 patient_id 列名/格式")

        # group by patient -> numpy [N,D]
        self.pid_to_feats = {}
        for pid, g in df_feat.groupby(self.pid_feat):
            arr = g[self.feat_cols].to_numpy(dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                self.pid_to_feats[str(pid)] = arr

        # pid -> label
        self.pid_to_label = dict(zip(df_lab[self.pid_lab].astype(str).tolist(),
                                     df_lab[label_col].astype(int).tolist()))

        # 最终 pid 列表：既有特征又有标签
        self.pids = [pid for pid in self.pid_to_label.keys() if pid in self.pid_to_feats]
        if len(self.pids) == 0:
            raise RuntimeError("对齐后可用患者为 0（特征/标签缺失），请检查数据")

        # 统计
        pos = sum(self.pid_to_label[pid] == 1 for pid in self.pids)
        print(f"[Dataset] patients={len(self.pids)}  pos={pos}  pos_rate={pos/len(self.pids):.3f}  feat_dim={self.feature_dim}")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        feats = self.pid_to_feats[pid]  # [N,D]
        y = int(self.pid_to_label[pid])

        # max_instances：train 随机采样，val 取前（对齐你原 dataset_patient.py 习惯）
        if self.max_instances > 0 and feats.shape[0] > self.max_instances:
            if self.train:
                sel = np.random.choice(feats.shape[0], self.max_instances, replace=False)
                feats = feats[sel]
            else:
                feats = feats[:self.max_instances]

        return torch.from_numpy(feats), torch.tensor(y, dtype=torch.long), pid


def collate_pad_mask(batch):
    """
    支持 batch_size>1：padding + mask（避免 padding 参与 attention/mean）
    returns:
      feats: [B, Nmax, D]
      mask:  [B, Nmax]  (True=有效实例)
      labels:[B]
      pids:  list[str]
    """
    feats_list, labels_list, pids = zip(*batch)
    B = len(feats_list)
    D = feats_list[0].shape[1]
    lengths = [f.shape[0] for f in feats_list]
    Nmax = max(lengths)

    feats = torch.zeros((B, Nmax, D), dtype=torch.float32)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)
    for i, f in enumerate(feats_list):
        n = f.shape[0]
        feats[i, :n, :] = f
        mask[i, :n] = True

    labels = torch.stack(labels_list, dim=0)
    return feats, mask, labels, list(pids)


# ------------------ Enhanced Attention MIL ------------------
class AttentionMIL(nn.Module):
    """
    增强版 Attention MIL：
      - 多头 self-attn (QKV)
      - gate
      - 残差
      - 最终用 mask-aware mean pooling
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """
        features: [B, N, D]
        mask:     [B, N]  True=有效
        """
        B, N, D = features.shape

        # LayerNorm（稳定）
        x = self.norm(features)

        # gate: [B,N]
        gate_w = self.gate(x).squeeze(-1)  # 0..1
        gate_w = gate_w * mask.float()     # padding 处强制 0

        # QKV: [B,H,N,Dh]
        Q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # attn_scores: [B,H,N,N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # mask keys：padding 的 key 不参与 softmax
        key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B,H,N,N]
        attn_output = torch.matmul(attn_weights, V)        # [B,H,N,Dh]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)  # [B,N,H*Dh]
        attn_output = self.out_proj(attn_output)           # [B,N,D]

        # 残差 + dropout
        weighted = x + self.dropout(attn_output)

        # 应用 gate（padding 强制为 0）
        weighted = weighted * gate_w.unsqueeze(-1)

        # mask-aware mean pooling
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B,1]
        pooled = (weighted * mask.unsqueeze(-1).float()).sum(dim=1) / denom  # [B,D]

        # 便于可视化：输出平均 head 的 N×N 注意力
        attn_mean = attn_weights.mean(dim=1)  # [B,N,N]
        return pooled, attn_mean, gate_w


class MILClassifier(nn.Module):
    """
    二分类 MIL：输出 2 类 logits
    prob1 = softmax(logits)[:,1]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, attn_hidden: int = 128):
        super().__init__()
        self.attention = AttentionMIL(input_dim=input_dim, hidden_dim=attn_hidden, n_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, 2)  # 2 classes
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        pooled, attn_mat, gate_w = self.attention(feats, mask)
        logits = self.classifier(pooled)
        return logits, pooled, attn_mat, gate_w


# ------------------ Metrics / Plots ------------------
def find_best_threshold(y_true, y_prob):
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def plot_cm(y_true, y_prob, th, out_path, title):
    y_pred = (y_prob >= th).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred0", "Pred1"],
                yticklabels=["True0", "True1"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_pr(y_true, y_prob, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{name}.png"))
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{name}.png"))
    plt.close()


# ------------------ Training ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_xlsx", type=str, required=True, help="特征表（xlsx/csv），含 patient_id + feat_*")
    parser.add_argument("--labels_xlsx", type=str, required=True, help="标签表（xlsx/csv），含 id + label_col")
    parser.add_argument("--label_col", type=str, default="内分泌代谢疾病")
    parser.add_argument("--patient_id_col_feat", type=str, default="patient_id")
    parser.add_argument("--patient_id_col_label", type=str, default="id")
    parser.add_argument("--feat_prefix", type=str, default="feat_")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_instances", type=int, default=0, help="<=0 全部实例；>0 截断/采样")
    parser.add_argument("--weighted_loss", action="store_true", help="单标签使用加权 CrossEntropyLoss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs_ce_mil_feat")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "training_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    # 先读标签表用于 split
    df_lab = load_table(args.labels_xlsx)
    if args.patient_id_col_label not in df_lab.columns:
        raise ValueError(f"标签表缺少列 '{args.patient_id_col_label}'")
    if args.label_col not in df_lab.columns:
        raise ValueError(f"标签表缺少标签列 '{args.label_col}'")

    df_lab[args.patient_id_col_label] = df_lab[args.patient_id_col_label].astype(str)
    df_lab[args.label_col] = df_lab[args.label_col].fillna(0)
    try:
        df_lab[args.label_col] = df_lab[args.label_col].astype(int)
    except Exception:
        df_lab[args.label_col] = df_lab[args.label_col].apply(lambda x: 1 if str(x).strip().lower() in ["1","yes","true","是","有","positive","患病"] else 0).astype(int)

    # 只保留在 features 里出现的患者（避免 split 出来但 dataset 里没特征）
    df_feat = load_table(args.features_xlsx)
    if args.patient_id_col_feat not in df_feat.columns:
        raise ValueError(f"特征表缺少列 '{args.patient_id_col_feat}'")
    feat_pids = set(df_feat[args.patient_id_col_feat].astype(str).unique().tolist())
    df_lab = df_lab[df_lab[args.patient_id_col_label].isin(feat_pids)].reset_index(drop=True)
    if len(df_lab) == 0:
        raise RuntimeError("标签表与特征表没有交集，请检查 patient_id / id 列")

    y = df_lab[args.label_col].values.astype(int)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("====== Training Log (CE MIL From XLSX) ======\n")
        f.write(f"Device: {device}\nSeed: {args.seed}\n")
        f.write(f"Features: {args.features_xlsx}\nLabels: {args.labels_xlsx}\n")
        f.write(f"Label: {args.label_col}\nFolds: {args.folds}\n\n")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df_lab)), y), start=1):
        print(f"\n===== Fold {fold}/{args.folds} =====")
        df_tr = df_lab.iloc[tr_idx].reset_index(drop=True)
        df_va = df_lab.iloc[va_idx].reset_index(drop=True)

        # 保存该折标签表（与原 train.py 类似）
        tr_path = os.path.join(args.out_dir, f"train_fold{fold}.csv")
        va_path = os.path.join(args.out_dir, f"valid_fold{fold}.csv")
        df_tr.to_csv(tr_path, index=False, encoding="utf-8-sig")
        df_va.to_csv(va_path, index=False, encoding="utf-8-sig")

        # Dataset
        train_ds = PatientBagFeatureDataset(
            features_table=args.features_xlsx,
            labels_table=tr_path,
            label_col=args.label_col,
            patient_id_col_feat=args.patient_id_col_feat,
            patient_id_col_label=args.patient_id_col_label,
            feat_prefix=args.feat_prefix,
            max_instances=args.max_instances,
            train=True
        )
        valid_ds = PatientBagFeatureDataset(
            features_table=args.features_xlsx,
            labels_table=va_path,
            label_col=args.label_col,
            patient_id_col_feat=args.patient_id_col_feat,
            patient_id_col_label=args.patient_id_col_label,
            feat_prefix=args.feat_prefix,
            max_instances=args.max_instances,
            train=False
        )

        in_dim = train_ds.feature_dim
        model = MILClassifier(input_dim=in_dim).to(device)

        # Loss：CrossEntropyLoss（可选加权）
        if args.weighted_loss:
            ytr = df_tr[args.label_col].values.astype(int)
            c0 = max(int((ytr == 0).sum()), 1)
            c1 = max(int((ytr == 1).sum()), 1)
            w0 = len(ytr) / (2.0 * c0)
            w1 = len(ytr) / (2.0 * c1)
            class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("CrossEntropyLoss weighted:", class_weights.detach().cpu().numpy().tolist())
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_pad_mask
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_pad_mask
        )

        best_auc = -1.0
        best_path = os.path.join(args.out_dir, f"best_fold{fold}.ckpt")

        for epoch in range(1, args.epochs + 1):
            # ---- train ----
            model.train()
            tr_loss_sum, tr_n = 0.0, 0
            for feats, mask, labels, _ in train_loader:
                feats = feats.to(device)
                mask = mask.to(device)
                labels = labels.to(device)  # [B]

                logits, _, _, _ = model(feats, mask)  # [B,2]
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tr_loss_sum += float(loss.item()) * labels.size(0)
                tr_n += labels.size(0)

            tr_loss = tr_loss_sum / max(tr_n, 1)

            # ---- valid ----
            model.eval()
            va_loss_sum, va_n = 0.0, 0
            all_ids, all_y, all_prob1 = [], [], []
            with torch.no_grad():
                for feats, mask, labels, pids in valid_loader:
                    feats = feats.to(device)
                    mask = mask.to(device)
                    labels = labels.to(device)

                    logits, _, _, _ = model(feats, mask)
                    loss = criterion(logits, labels)

                    prob1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # [B]
                    va_loss_sum += float(loss.item()) * labels.size(0)
                    va_n += labels.size(0)

                    all_prob1.append(prob1)
                    all_y.append(labels.detach().cpu().numpy())
                    all_ids.extend(pids)

            all_prob1 = np.concatenate(all_prob1, axis=0)
            all_y = np.concatenate(all_y, axis=0).astype(int)
            va_loss = va_loss_sum / max(va_n, 1)

            # AUC
            try:
                auc = roc_auc_score(all_y, all_prob1) if len(np.unique(all_y)) >= 2 else np.nan
            except Exception:
                auc = np.nan

            # 阈值搜索（F1 最优）
            th, best_f1 = find_best_threshold(all_y, all_prob1)
            y_pred = (all_prob1 >= th).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(all_y, y_pred, average="binary", zero_division=0)
            acc = accuracy_score(all_y, y_pred)
            cm = confusion_matrix(all_y, y_pred, labels=[0, 1]).tolist()

            # 每 epoch 保存 preds（对齐你原 train.py 习惯）
            prob_dir = os.path.join(args.out_dir, f"fold{fold}_probs")
            os.makedirs(prob_dir, exist_ok=True)
            pred_df = pd.DataFrame({
                "id": all_ids,
                f"{args.label_col}_y": all_y.astype(int),
                f"{args.label_col}_prob": all_prob1.astype(float),
            })
            pred_path = os.path.join(prob_dir, f"epoch{epoch}_preds.csv")
            pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

            # 每 epoch 保存 CM
            metric_dir = os.path.join(args.out_dir, f"fold{fold}_metrics")
            os.makedirs(metric_dir, exist_ok=True)
            plot_cm(
                all_y, all_prob1, th,
                out_path=os.path.join(metric_dir, f"cm_{args.label_col}.png"),
                title=f"CM - {args.label_col} (Fold {fold}, Epoch {epoch}, Th={th:.2f})"
            )
            if epoch == args.epochs:
                plot_roc_pr(all_y, all_prob1, os.path.join(metric_dir, "final_epoch_curves"), args.label_col)

            # log
            auc_s = "nan" if np.isnan(auc) else f"{auc:.4f}"
            print(f"Fold {fold} | Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, AUC={auc_s}, F1={f1:.4f}, Th={th:.2f}")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Fold {fold} | Epoch {epoch}:\n")
                f.write(f"  train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}\n")
                f.write(f"  auc={auc_s}, precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}, acc={acc:.4f}, Th={th:.2f}, cm={cm}\n")
                f.write(f"  preds_csv={pred_path}\n\n")

            scheduler.step()

            # 保存最优（按 AUC）
            if not np.isnan(auc) and auc > best_auc:
                best_auc = float(auc)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "in_dim": in_dim,
                        "label_col": args.label_col,
                        "fold": fold,
                        "seed": args.seed,
                        "best_auc": best_auc,
                    },
                    best_path
                )
                print(f"  -> Save best: {best_path} (best_auc={best_auc:.4f})")

        print(f"Fold {fold} finished. best_auc={best_auc:.4f}")

    print("训练完成。输出目录：", args.out_dir)
    print("训练日志：", log_path)


if __name__ == "__main__":
    main()
"""
python train_ce_mil_from_xlsx.py \
  --features_xlsx ../features01.xlsx \
  --labels_xlsx ../../labels.xlsx \
  --label_col 内分泌代谢疾病 \
  --epochs 30 \
  --folds 4 \
  --batch_size 1 \
  --max_instances 0 \
  --out_dir outputs_attention_feat01

"""