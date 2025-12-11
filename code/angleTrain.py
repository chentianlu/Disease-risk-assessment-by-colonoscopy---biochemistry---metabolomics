# train.py
import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # pip install iterative-stratification

import matplotlib.pyplot as plt
import seaborn as sns

from dataset_patient import PatientBagDataset
from mil_train import PatientMILMultiLabel
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei']  # 任选一个你系统有的
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 工具函数 ==========

def load_labels_table(path: str) -> pd.DataFrame:
    """
    智能读取标签表：
    - 如果是 .xlsx/.xls，用 read_excel
    - 如果是 .csv，依次尝试 utf-8 / gbk / ansi / latin1
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        print(f"✅ 使用 Excel 方式读取标签: {path}")
        return pd.read_excel(path)
    else:
        for enc in ["utf-8", "gbk", "ansi", "latin1"]:
            try:
                print(f"尝试用编码 {enc} 读取 {path} ...")
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                print(f"编码 {enc} 失败，继续尝试下一个编码。")
                continue
        raise UnicodeDecodeError(
            "all", b"", 0, 1,
            f"无法用 utf-8/gbk/ansi/latin1 读取 {path}，请确认文件编码。"
        )


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """
    计算每个标签的 pos_weight，并做平滑。
    这里采用更温和的方案：先按 n/p 算，再做 sqrt + 归一化，最后 clip 到 [0.5, 3].
    """
    M, L = labels.shape
    raw_list = []
    for j in range(L):
        p = labels[:, j].sum()
        n = M - p
        raw = (n + 1e-6) / (p + 1e-6)     # n/p
        raw_list.append(raw)

    raw_arr = np.array(raw_list, dtype=np.float32)

    # 先开方（减小极端值），再除以平均值让整体均值≈1
    w = np.sqrt(raw_arr)

    # 最后再做一个比较窄的 clip，比如 [0.5, 3.0]
    w = np.clip(w, 0.5, 3.0)

    print("pos_weight per label:", w)

    return torch.tensor(w, dtype=torch.float32)




def describe_split(df_part: pd.DataFrame, label_cols, name: str):
    """
    打印每个 split 的阳性率，便于检查多标签分层是否平衡。
    """
    tot = len(df_part)
    rates = {c: float(df_part[c].mean()) for c in label_cols}
    msg = f"[{name}] n={tot}  " + "  ".join([f"{k}:pos_rate={v:.3f}" for k, v in rates.items()])
    print(msg)
    return msg


def compute_epoch_metrics(all_labels: np.ndarray,
                          all_probs: np.ndarray,
                          label_cols,
                          threshold: float = 0.5,
                          search_best_threshold: bool = False):
    """
    计算每个标签的 AUC, precision, recall, f1, accuracy, confusion matrix。
    如果 search_best_threshold=True，则对每个标签在若干候选阈值上搜索 F1 最优阈值。

    返回:
      mean_auc, per_label_auc_list, metrics_dict
    metrics_dict[label_name] = {
        'auc', 'precision', 'recall', 'f1', 'acc',
        'cm' (2x2 list [[TN,FP],[FN,TP]]),
        'threshold' : 使用的阈值（若 search_best_threshold=True，则为该标签的最佳阈值）
    }
    """
    L = all_labels.shape[1]
    aucs = []
    metrics = {}
    for j in range(L):
        name = label_cols[j]
        y_true = all_labels[:, j]
        y_prob = all_probs[:, j]

        # ===== AUC =====
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan

        # ===== 是否为该标签搜索最佳阈值 =====
        if search_best_threshold:
            best_f1 = -1.0
            best_t = threshold
            best_p = 0.0
            best_r = 0.0

            # 在 0.1 ~ 0.9 之间扫一圈阈值（你可以按需调整颗粒度）
            for t in np.linspace(0.1, 0.9, 17):
                y_pred_tmp = (y_prob >= t).astype(int)
                p_tmp, r_tmp, f1_tmp, _ = precision_recall_fscore_support(
                    y_true, y_pred_tmp,
                    average="binary",
                    zero_division=0
                )
                if f1_tmp > best_f1:
                    best_f1 = f1_tmp
                    best_t = t
                    best_p = p_tmp
                    best_r = r_tmp

            # 用最佳阈值重新得到预测标签
            y_pred = (y_prob >= best_t).astype(int)
            precision = best_p
            recall = best_r
            f1 = best_f1
            used_t = best_t
        else:
            # 固定使用传入的 threshold（例如 0.5 或 0.3）
            y_pred = (y_prob >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred,
                average="binary",
                zero_division=0
            )
            used_t = threshold

        # ===== Accuracy & Confusion Matrix =====
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        aucs.append(auc)
        metrics[name] = {
            "auc": float(auc) if not np.isnan(auc) else None,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "acc": float(acc),
            "cm": cm.tolist(),
            "threshold": float(used_t),   # 记录这个标签用的阈值
        }

    mean_auc = float(np.nanmean(aucs))
    return mean_auc, aucs, metrics



def plot_roc_pr_curves(all_labels: np.ndarray,
                       all_probs: np.ndarray,
                       label_names,
                       out_dir: str):
    """
    为每个标签绘制 ROC 和 PR 曲线，并在 ROC 图中加入 AUC 值。
    """
    os.makedirs(out_dir, exist_ok=True)

    for i, name in enumerate(label_names):
        yt = all_labels[:, i]
        yp_prob = all_probs[:, i]

        # 如果标签没有正例或负例，无法计算 ROC/PR
        if len(np.unique(yt)) < 2:
            print(f"[WARN] Label '{name}' 在验证集中只有单一类别（无法计算 ROC/AUC），跳过。")
            continue

        # ========== ROC 曲线 + AUC ========== 
        try:
            fpr, tpr, _ = roc_curve(yt, yp_prob)
            auc_val = roc_auc_score(yt, yp_prob)

            plt.figure(figsize=(5,4))
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

        # ========== PR 曲线（可选加入 AUPR） ==========
        try:
            precision, recall, _ = precision_recall_curve(yt, yp_prob)
            plt.figure(figsize=(5,4))
            plt.plot(recall, precision, lw=2)

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve - {name}")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pr_{name}.png"))
            plt.close()
        except Exception as e:
            print(f"[WARN] PR 绘制失败: {name}, err={e}")



def plot_confusion_matrices(all_labels: np.ndarray,
                            all_probs: np.ndarray,
                            label_names,
                            metrics: dict,
                            out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for i, name in enumerate(label_names):
        yt = all_labels[:, i]
        yp_prob = all_probs[:, i]

        t = metrics[name].get("threshold", 0.5)  # ★ 用 compute_epoch_metrics 里保存的阈值
        yp = (yp_prob >= t).astype(int)

        cm = confusion_matrix(yt, yp, labels=[0, 1])

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"]
        )
        plt.title(f"Confusion Matrix - {name}\n(t={t:.2f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{name}.png"))
        plt.close()



# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--labels_csv', type=str, default='../labels.csv')
    parser.add_argument('--label_cols', nargs='+', required=True,
                        help='多标签列名，例如: 消化系统疾病 内分泌代谢疾病 泌尿生殖系统疾病 血液及造血器官疾病和涉及免疫机制的某些疾患')
    parser.add_argument('--max_images', type=int, default=0,
                        help='<=0 使用全部图像；>0 时最多取这么多张图')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='使用全部图像时建议 batch_size=1（每个 batch 一个病人）')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='outputs_angle')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='efficientnet_b0')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='是否使用离线预训练权重(1/0)')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='离线权重 .pth 文件路径')
    parser.add_argument('--log_file', type=str, default='training_log.txt',
                        help='训练日志文件名（保存在 out_dir 下）')
    parser.add_argument('--freeze_encoder', type=int, default=1,
                        help='是否冻结编码器权重 (1/0)')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_file)

    # ========== 1. 读取标签 ==========
    if args.labels_csv.lower().endswith(".xlsx") or args.labels_csv.lower().endswith(".xls"):
        df = pd.read_excel(args.labels_csv)
    else:
        df = load_labels_table(args.labels_csv)

    if "id" not in df.columns:
        raise ValueError("标签文件中必须包含一列名为 'id' 的患者编号。")

    label_cols = args.label_cols
    for col in label_cols:
        if col not in df.columns:
            raise ValueError(f"标签列 '{col}' 不在标签文件中。")

    # 多标签矩阵 [M, L]，用于多标签分层划分
    # 多标签矩阵 [M, L]
    y_all = df[label_cols].values.astype('int')
    M, L = y_all.shape

    # ========== 2. 按标签数量选择合适的 K 折 ==========
    if L == 1:
        # ★ 单标签（二分类或多分类）：用普通 StratifiedKFold
        print("🔁 检测到单标签任务，使用 StratifiedKFold 做分层划分。")
        splitter = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=42
        )
        y_target = y_all[:, 0]    # [M]
    else:
        # ★ 多标签任务：继续用 MultilabelStratifiedKFold
        print("🔁 检测到多标签任务，使用 MultilabelStratifiedKFold 做分层划分。")
        splitter = MultilabelStratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=42
        )
        y_target = y_all          # [M, L]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("✅ Using device:", device)

    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write("====== Training Log ======\n")
        lf.write(f"Device: {device}\n")
        lf.write(f"Label columns: {label_cols}\n")
        lf.write(f"Folds: {args.folds}\n\n")

    fold = 0

    for tr_idx, va_idx in splitter.split(np.zeros(len(df)), y_target):
        fold += 1
        print(f"\n===== Fold {fold}/{args.folds} =====")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        msg_tr = describe_split(df_tr, label_cols, f"Fold{fold}-Train")
        msg_va = describe_split(df_va, label_cols, f"Fold{fold}-Valid")

        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f"===== Fold {fold}/{args.folds} =====\n")
            lf.write(msg_tr + "\n")
            lf.write(msg_va + "\n")

        # 保存本折的标签 csv（可选）
        tr_csv = os.path.join(args.out_dir, f'train_fold{fold}.csv')
        va_csv = os.path.join(args.out_dir, f'valid_fold{fold}.csv')
        df_tr.to_csv(tr_csv, index=False)
        df_va.to_csv(va_csv, index=False)

        # max_images<=0: 使用全部图像
        max_images = args.max_images if args.max_images > 0 else 0

        train_ds = PatientBagDataset(
            args.data_root,
            tr_csv,
            label_cols,
            max_images=max_images,
            train=True
        )
        valid_ds = PatientBagDataset(
            args.data_root,
            va_csv,
            label_cols,
            max_images=max_images,
            train=False
        )

        def collate(batch):
            # batch_size 建议为 1，这样支持每个病人 N 不同
            bags, ys, pids, paths = zip(*batch)
            # print("当前 batch 的患者 ID:", pids[0])
            bags = bags[0].unsqueeze(0)  # [1, N, 3, H, W]
            ys = torch.stack(ys, dim=0)  # [B, L]
            return bags, ys, pids, paths

        tr_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,  # 推荐 1
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

        # ========== 3. 构建模型 ==========
        freeze_flag = bool(args.freeze_encoder)
        model = PatientMILMultiLabel(
            n_labels=len(label_cols),
            encoder_name=args.encoder,
            pretrained=bool(args.pretrained),
            weights_path=args.weights_path,
            freeze_encoder=freeze_flag
        ).to(device)


        # 只优化可训练参数（encoder 已被冻结时不会更新）
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        # 计算本折 pos_weight
        pos_weight = compute_pos_weight(df_tr[label_cols].values.astype('float32')).to(device)
        print(f"[Fold {fold}] pos_weight = {pos_weight.cpu().numpy()}")  # ★ 新增：控制台打印
        # ★ 新增：写到日志文件
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f"[Fold {fold}] pos_weight = {pos_weight.cpu().numpy().tolist()}\n")

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = -1.0
        best_path = os.path.join(args.out_dir, f'best_fold{fold}.ckpt')

        for epoch in range(1, args.epochs + 1):
            # ----- 训练 -----
            model.train()
            tr_loss_sum = 0.0
            n_tr_samples = 0

            for bags, ys, _, _ in tr_loader:
                bags = bags.to(device)
                # print(f"正在训练,正在打印bags的形状: {bags.shape}")
                # print(f"正在打印ys的形状: {ys.shape}")  
                ys = ys.to(device)
                logits, _, _ = model(bags)
                # print("正在打印logits的值:", logits)
                # print("正在打印ys的值:", ys)
                loss = criterion(logits, ys)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                B = ys.size(0)
                tr_loss_sum += loss.item() * B
                n_tr_samples += B

            tr_loss = tr_loss_sum / max(n_tr_samples, 1)

            # ----- 验证 -----
            model.eval()
            all_logits = []
            all_labels = []
            all_ids = []
            with torch.no_grad():
                for bags, ys, pids, _ in va_loader:
                    bags = bags.to(device)
                    ys = ys.to(device)
                    logits, _, _ = model(bags)
                    all_logits.append(logits.sigmoid().cpu().numpy())
                    all_labels.append(ys.cpu().numpy())
                    all_ids.extend(list(pids))
            all_logits = np.concatenate(all_logits, axis=0)  # [M_val, L]
            all_labels = np.concatenate(all_labels, axis=0)  # [M_val, L]

            mean_auc, per_label_aucs, metrics = compute_epoch_metrics(
                all_labels, all_logits, label_cols,
                threshold=0.5,                 # 初始值
                search_best_threshold=True     
            )
            # ====== 保存当前 epoch 的病人预测表 ======
            # 目录：outputs/fold{fold}_probs/epoch{epoch}_preds.csv
            
            prob_dir = os.path.join(args.out_dir, f"fold{fold}_probs")
            os.makedirs(prob_dir, exist_ok=True)
            prob_path = os.path.join(prob_dir, f"epoch{epoch}_preds.csv")

            # 构造 DataFrame：id + 每个标签的 y 和 prob
            df_epoch = pd.DataFrame()
            df_epoch["id"] = all_ids

            for j, name in enumerate(label_cols):
                df_epoch[f"{name}_y"] = all_labels[:, j]
                df_epoch[f"{name}_prob"] = all_logits[:, j]

            df_epoch.to_csv(prob_path, index=False, encoding="utf-8-sig")
            print(f"[Fold {fold}][Epoch {epoch}] 验证集预测结果已保存到: {prob_path}")
            # ======  绘制 ROC + PR 曲线 ======
            if epoch == args.epochs:
                curve_dir = os.path.join(args.out_dir, f"fold{fold}_metrics", "final_epoch_curves")
                plot_roc_pr_curves(
                    all_labels,
                    all_logits,
                    label_cols,
                    out_dir=curve_dir
                )


            # 画混淆矩阵图像
            cm_dir = os.path.join(args.out_dir, f"fold{fold}_metrics")
            plot_confusion_matrices(
                all_labels,
                all_logits,
                label_cols,
                metrics,        # ✅ 这里传 metrics，而不是 threshold
                cm_dir
            )

            # 控制台打印更详细的指标
            print("\n===== Per-label Metrics (Fold {}, Epoch {}) =====".format(fold, epoch))
            for name in label_cols:
                m = metrics[name]
                auc_val = m["auc"]
                auc_str = f"{auc_val:.3f}" if auc_val is not None else "nan"
                print(f"{name}: AUC={auc_str}, "
                      f"P={m['precision']:.3f}, R={m['recall']:.3f}, "
                      f"F1={m['f1']:.3f}, Acc={m['acc']:.3f}, "
                      f"Th={m['threshold']:.3f}"        # ★ 新增：打印阈值
                    )
            print("============================================\n")

            # 日志写入
            print(f"Fold {fold} | Epoch {epoch}: "
                  f"train_loss={tr_loss:.4f}, val_meanAUC={mean_auc:.4f}, per_label={per_label_aucs}")

            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(f"Fold {fold} | Epoch {epoch}:\n")
                lf.write(f"  train_loss={tr_loss:.4f}\n")
                lf.write(f"  val_meanAUC={mean_auc:.4f}\n")
                lf.write(f"  per_label AUC={per_label_aucs}\n")
                for name in label_cols:
                    m = metrics[name]
                    auc_val = m["auc"]
                    auc_str = f"{auc_val:.4f}" if auc_val is not None else "nan"
                    lf.write(
                        f"  [{name}] auc={auc_str}, "
                        f"precision={m['precision']:.4f}, recall={m['recall']:.4f}, "
                        f"f1={m['f1']:.4f}, acc={m['acc']:.4f}, "
                         f"th={m['threshold']:.4f}, " 
                        f"cm={m['cm']}\n"
                    )
                lf.write("\n")

            scheduler.step()

            # 保存当前折中 AUC 最好的模型
            if mean_auc > best_auc:
                best_auc = mean_auc
                torch.save(
                    {
                        'model': model.state_dict(),
                        'label_cols': label_cols
                    },
                    best_path
                )
                print(f"  -> Save best to {best_path}")

        print(f"Fold {fold} finished. Best mean AUC={best_auc:.4f}")
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f"Fold {fold} best_meanAUC={best_auc:.4f}\n\n")

    print("训练完成。每折最优权重见 outputs_angle/ 目录，训练日志见", log_path)


if __name__ == '__main__':
    main()



"""
CUDA_VISIBLE_DEVICES=1 python angleTrain.py \
  --data_root ../data \
  --labels_csv ./labels.xlsx \
  --label_cols 内分泌代谢疾病 \
  --epochs 30 \
  --batch_size 1 \
  --max_images -1 \
  --folds 4 \
  --num_workers 0 \
  --encoder vit_base_patch14_dinov2 \
  --pretrained 1 \
  --weights_path ../weight/dinov2_vitb14_pretrain.pth \
  --freeze_encoder 1


"""