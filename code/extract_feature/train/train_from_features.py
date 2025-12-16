# train_from_features.py
import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import matplotlib
matplotlib.use("Agg")  # 服务器无显示器也能画图
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_patient_features import PatientBagFeatureDataset
from mil_train_features import PatientMILMultiLabelFromFeatures


# ------------------ Reproducibility ------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------ IO helpers ------------------
def load_labels_table(path: str) -> pd.DataFrame:
    """
    csv 多编码兜底读取（xlsx 在 main 里直接 pd.read_excel）
    """
    for enc in ["utf-8", "gbk", "ansi", "latin1"]:
        try:
            print(f"尝试用编码 {enc} 读取 {path} ...")
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            print(f"编码 {enc} 失败，继续尝试下一个编码。")
    raise UnicodeDecodeError("all", b"", 0, 1, f"无法读取: {path}")


def describe_split(df_part: pd.DataFrame, label_cols, name: str) -> str:
    tot = len(df_part)
    rates = {c: float(df_part[c].mean()) for c in label_cols}
    msg = f"[{name}] n={tot}  " + "  ".join([f"{k}:pos_rate={v:.3f}" for k, v in rates.items()])
    print(msg)
    return msg


# ------------------ Metrics helpers ------------------
def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """
    对齐你最新 train.py：
      raw = n/p
      w = sqrt(raw)
      w = w / mean(w)
      w = clip(w, 0.5, 3.0)
    """
    M, L = labels.shape
    raw_list = []
    for j in range(L):
        p = labels[:, j].sum()
        n = M - p
        raw = (n + 1e-6) / (p + 1e-6)
        raw_list.append(raw)

    raw_arr = np.array(raw_list, dtype=np.float32)
    w = np.sqrt(raw_arr)
    
    w = np.clip(w, 0.5, 3.0)

    print("pos_weight per label:", w.tolist())
    return torch.tensor(w, dtype=torch.float32)


def compute_epoch_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    label_cols,
    threshold: float = 0.5,
    search_best_threshold: bool = True
):
    """
    返回：
      mean_auc, per_label_auc_list, metrics_dict

    metrics_dict[label] = {
      auc, precision, recall, f1, acc,
      cm=[[TN,FP],[FN,TP]],
      threshold=Th
    }
    """
    L = all_labels.shape[1]
    aucs = []
    metrics = {}

    for j in range(L):
        name = label_cols[j]
        y_true = all_labels[:, j].astype(int)
        y_prob = all_probs[:, j].astype(float)

        # AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan

        # 阈值搜索：F1 最优（对齐你最新 train.py 的“每标签 Th”思路）
        used_t = threshold
        if search_best_threshold:
            best = {"f1": -1.0, "t": threshold, "p": 0.0, "r": 0.0}
            for t in np.linspace(0.1, 0.9, 17):
                y_pred_tmp = (y_prob >= t).astype(int)
                p_tmp, r_tmp, f1_tmp, _ = precision_recall_fscore_support(
                    y_true, y_pred_tmp, average="binary", zero_division=0
                )
                if f1_tmp > best["f1"]:
                    best = {"f1": f1_tmp, "t": float(t), "p": float(p_tmp), "r": float(r_tmp)}
            used_t = best["t"]
            precision, recall, f1 = best["p"], best["r"], best["f1"]
        else:
            y_pred_tmp = (y_prob >= used_t).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_tmp, average="binary", zero_division=0
            )

        y_pred = (y_prob >= used_t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        aucs.append(auc)
        metrics[name] = {
            "auc": None if np.isnan(auc) else float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "acc": float(acc),
            "cm": cm.tolist(),
            "threshold": float(used_t),
        }

    mean_auc = float(np.nanmean(aucs)) if len(aucs) else float("nan")
    return mean_auc, aucs, metrics


def plot_roc_pr_curves(all_labels: np.ndarray, all_probs: np.ndarray, label_names, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(label_names):
        yt = all_labels[:, i].astype(int)
        yp = all_probs[:, i].astype(float)

        if len(np.unique(yt)) < 2:
            print(f"[WARN] Label '{name}' 在验证集中只有单一类别，跳过 ROC/PR。")
            continue

        # ROC
        try:
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_val = roc_auc_score(yt, yp)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - {name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"roc_{name}.png"))
            plt.close()
        except Exception as e:
            print(f"[WARN] ROC 绘制失败: {name}, err={e}")

        # PR
        try:
            precision, recall, _ = precision_recall_curve(yt, yp)
            plt.figure(figsize=(5, 4))
            plt.plot(recall, precision, lw=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve - {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pr_{name}.png"))
            plt.close()
        except Exception as e:
            print(f"[WARN] PR 绘制失败: {name}, err={e}")


def plot_confusion_matrices(all_labels: np.ndarray, all_probs: np.ndarray, label_names, metrics: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(label_names):
        yt = all_labels[:, i].astype(int)
        yp = all_probs[:, i].astype(float)
        th = float(metrics[name].get("threshold", 0.5))
        y_pred = (yp >= th).astype(int)

        cm = confusion_matrix(yt, y_pred, labels=[0, 1])

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"]
        )
        plt.title(f"Confusion Matrix - {name}\n(Th={th:.2f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{name}.png"))
        plt.close()


# ------------------ Main training ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_xlsx", type=str, required=True, help="tile_extract_to_xlsx.py 导出的特征表（xlsx/csv）")
    parser.add_argument("--labels_csv", type=str, required=True, help="标签表（xlsx/csv），包含 id + 多标签列")
    parser.add_argument("--label_cols", nargs="+", required=True)
    parser.add_argument("--max_instances", type=int, default=0, help="<=0 用全部实例；>0 截断/采样")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1, help="建议 1（每个 batch 一个病人）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs_feat")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--d_hidden_attn", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.batch_size != 1:
        raise ValueError("当前实现为了严格对齐你原版流程，建议 batch_size=1（每个 batch 一个患者）。")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "training_log.txt")

    # ===== 读标签（对齐你最新 train.py：xlsx 直接 read_excel，否则多编码读取）=====
    ext = os.path.splitext(args.labels_csv)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(args.labels_csv)
    else:
        df = load_labels_table(args.labels_csv)

    if "id" not in df.columns:
        raise ValueError("标签文件中必须包含一列名为 'id' 的患者编号。")
    for col in args.label_cols:
        if col not in df.columns:
            raise ValueError(f"标签列 '{col}' 不在标签文件中。")

    # 规范化：id 转 str，标签转 int(0/1)
    df["id"] = df["id"].astype(str)
    df[args.label_cols] = df[args.label_cols].fillna(0).astype(int)

    y_all = df[args.label_cols].values.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("====== Training Log (From Features) ======\n")
        lf.write(f"Device: {device}\n")
        lf.write(f"Seed: {args.seed}\n")
        lf.write(f"Features: {args.features_xlsx}\n")
        lf.write(f"Labels: {args.labels_csv}\n")
        lf.write(f"Label columns: {args.label_cols}\n")
        lf.write(f"Folds: {args.folds}\n\n")

    # collate：batch_size=1，保证 [1, N, D]
    def collate(batch):
        feats, ys, pids, paths = zip(*batch)
        feats = feats[0].unsqueeze(0)     # [1, N, D]
        ys = torch.stack(ys, dim=0)       # [1, L]
        return feats, ys, pids, paths

    # ====== K折分层：单标签用 StratifiedKFold，多标签用 MultilabelStratifiedKFold ======
    # 确保 y_all 至少二维，避免被当成 binary
    if y_all.ndim == 1:
        y_all = y_all.reshape(-1, 1)

    if y_all.shape[1] == 1:
        # 单标签（二分类）分层K折
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = skf.split(np.zeros(len(df)), y_all[:, 0])
    else:
        # 多标签分层K折
        mskf = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = mskf.split(np.zeros(len(df)), y_all)

    fold = 0
    for tr_idx, va_idx in split_iter:
        fold += 1
        print(f"\n===== Fold {fold}/{args.folds} =====")

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        msg_tr = describe_split(df_tr, args.label_cols, f"Fold{fold}-Train")
        msg_va = describe_split(df_va, args.label_cols, f"Fold{fold}-Valid")

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"===== Fold {fold}/{args.folds} =====\n")
            lf.write(msg_tr + "\n")
            lf.write(msg_va + "\n")

        # 保存该折标签表（对齐你 train.py 的习惯）
        tr_csv = os.path.join(args.out_dir, f"train_fold{fold}.csv")
        va_csv = os.path.join(args.out_dir, f"valid_fold{fold}.csv")
        df_tr.to_csv(tr_csv, index=False, encoding="utf-8-sig")
        df_va.to_csv(va_csv, index=False, encoding="utf-8-sig")

        train_ds = PatientBagFeatureDataset(
            features_table=args.features_xlsx,
            labels_csv=tr_csv,
            label_cols=args.label_cols,
            max_instances=args.max_instances,
            train=True
        )
        valid_ds = PatientBagFeatureDataset(
            features_table=args.features_xlsx,
            labels_csv=va_csv,
            label_cols=args.label_cols,
            max_instances=args.max_instances,
            train=False
        )

        tr_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate
        )
        va_loader = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate
        )

        # ===== 模型：in_dim 自动从特征列数得到 =====
        in_dim = train_ds.feature_dim
        model = PatientMILMultiLabelFromFeatures(
            in_dim=in_dim,
            n_labels=len(args.label_cols),
            d_hidden_attn=args.d_hidden_attn,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        pos_weight = compute_pos_weight(df_tr[args.label_cols].values.astype(np.float32)).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = -1.0
        best_path = os.path.join(args.out_dir, f"best_fold{fold}.ckpt")

        for epoch in range(1, args.epochs + 1):
            # ---- train ----
            model.train()
            tr_loss_sum = 0.0
            n_tr = 0

            for feats, ys, _, _ in tr_loader:
                feats = feats.to(device)  # [1, N, D]
                ys = ys.to(device)        # [1, L]

                logits, _, _ = model(feats)
                loss = criterion(logits, ys)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tr_loss_sum += float(loss.item())
                n_tr += 1

            tr_loss = tr_loss_sum / max(n_tr, 1)

            # ---- valid ----
            model.eval()
            all_probs, all_labels, all_ids = [], [], []
            with torch.no_grad():
                for feats, ys, pids, _ in va_loader:
                    feats = feats.to(device)
                    ys = ys.to(device)

                    logits, _, _ = model(feats)
                    prob = logits.sigmoid().cpu().numpy()   # [1, L]

                    all_probs.append(prob)
                    all_labels.append(ys.cpu().numpy())
                    all_ids.append(str(pids[0]))

            all_probs = np.concatenate(all_probs, axis=0)    # [M, L]
            all_labels = np.concatenate(all_labels, axis=0)  # [M, L]

            # ===== 每 epoch 保存验证集预测表（完全对齐你最新 train.py）=====
            prob_dir = os.path.join(args.out_dir, f"fold{fold}_probs")
            os.makedirs(prob_dir, exist_ok=True)
            pred_df = pd.DataFrame({"id": all_ids})
            for j, name in enumerate(args.label_cols):
                pred_df[f"{name}_y"] = all_labels[:, j].astype(int)
                pred_df[f"{name}_prob"] = all_probs[:, j].astype(float)
            pred_path = os.path.join(prob_dir, f"epoch{epoch}_preds.csv")
            pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

            # ===== 指标：阈值搜索 + Th 记录 =====
            mean_auc, per_label_aucs, metrics = compute_epoch_metrics(
                all_labels, all_probs, args.label_cols,
                threshold=0.5,
                search_best_threshold=True
            )

            # 每轮画 CM（用该标签的 Th）
            cm_dir = os.path.join(args.out_dir, f"fold{fold}_metrics")
            plot_confusion_matrices(all_labels, all_probs, args.label_cols, metrics, cm_dir)

            # 最后一轮画 ROC/PR
            if epoch == args.epochs:
                curve_dir = os.path.join(args.out_dir, f"fold{fold}_metrics", "final_epoch_curves")
                plot_roc_pr_curves(all_labels, all_probs, args.label_cols, curve_dir)

            # 控制台打印（含 Th）
            print(f"Fold {fold} | Epoch {epoch}: train_loss={tr_loss:.4f}, val_meanAUC={mean_auc:.4f}")
            for name in args.label_cols:
                m = metrics[name]
                auc_v = m["auc"]
                auc_s = f"{auc_v:.4f}" if auc_v is not None else "nan"
                th = m.get("threshold", 0.5)
                print(
                    f"  [{name}] auc={auc_s}, P={m['precision']:.4f}, R={m['recall']:.4f}, "
                    f"F1={m['f1']:.4f}, Acc={m['acc']:.4f}, Th={th:.2f}, cm={m['cm']}"
                )

            # 写日志（含 Th）
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"Fold {fold} | Epoch {epoch}:\n")
                lf.write(f"  train_loss={tr_loss:.4f}\n")
                lf.write(f"  val_meanAUC={mean_auc:.4f}\n")
                lf.write(f"  per_label AUC={per_label_aucs}\n")
                lf.write(f"  preds_csv={pred_path}\n")
                for name in args.label_cols:
                    m = metrics[name]
                    auc_v = m["auc"]
                    auc_s = f"{auc_v:.4f}" if auc_v is not None else "nan"
                    th = m.get("threshold", 0.5)
                    lf.write(
                        f"  [{name}] auc={auc_s}, precision={m['precision']:.4f}, "
                        f"recall={m['recall']:.4f}, f1={m['f1']:.4f}, acc={m['acc']:.4f}, "
                        f"Th={th:.2f}, cm={m['cm']}\n"
                    )
                lf.write("\n")

            scheduler.step()

            # 保存最优
            if mean_auc > best_auc:
                best_auc = mean_auc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "label_cols": args.label_cols,
                        "in_dim": in_dim,
                        "seed": args.seed,
                        "fold": fold,
                        "best_mean_auc": best_auc,
                    },
                    best_path
                )
                print(f"  -> Save best to {best_path}")

        print(f"Fold {fold} finished. Best mean AUC={best_auc:.4f}")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"Fold {fold} best_meanAUC={best_auc:.4f}\n\n")

    print("训练完成。每折最优权重见：", args.out_dir)
    print("训练日志：", log_path)


if __name__ == "__main__":
    main()

"""
python train_from_features.py \
  --features_xlsx ../tile_features.xlsx \
  --labels_csv ../../labels.xlsx \
  --label_cols 内分泌代谢疾病  \
  --epochs 30 \
  --batch_size 1 \
  --max_instances 0 \
  --folds 4 \
  --num_workers 0 \
  --out_dir outputs_feat

"""