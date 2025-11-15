import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_patient import PatientBagDataset



def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """ 计算每个标签的 pos_weight（用于 BCEWithLogitsLoss 处理类不平衡）。 labels: [M, L] """
    M, L = labels.shape
    pos_w = []
    for j in range(L):
        p = labels[:, j].sum()
        n = M - p
        w = (n + 1e-6) / (p + 1e-6)
        pos_w.append(w)
    return torch.tensor(pos_w, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=int, default=1, help='是否使用离线预训练权重(1/0)')
    parser.add_argument('--weights_path', type=str, default=None, help='离线权重 .pth 文件路径')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--labels_csv', type=str, default='labels.csv')
    parser.add_argument('--label_cols', nargs='+', required=True, help='多标签列名，例如: diabetes disease_B')
    parser.add_argument('--max_images', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)  # 每个 batch 是2个患者
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='efficientnet_b0')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取标签，做分层（对多标签：简单使用第一列分层；若需要更稳健，可自定义多标签分层策略）
    df = pd.read_excel(args.labels_csv)
    y_all = df[args.label_cols].values.astype('float32')
    first_label = df[args.label_cols[0]].values.astype('int')

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold = 0
    for tr_idx, va_idx in skf.split(np.zeros(len(first_label)), first_label):
        fold += 1
        print(f"===== Fold {fold}/{args.folds} =====")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        # 保存临时 CSV，便于 Dataset 读取
        tr_csv = os.path.join(args.out_dir, f'train_fold{fold}.csv')
        va_csv = os.path.join(args.out_dir, f'valid_fold{fold}.csv')
        df_tr.to_csv(tr_csv, index=False)
        df_va.to_csv(va_csv, index=False)

        train_ds = PatientBagDataset(args.data_root, tr_csv, args.label_cols,
                                     max_images=args.max_images, train=True)
        valid_ds = PatientBagDataset(args.data_root, va_csv, args.label_cols,
                                     max_images=args.max_images, train=False)

        def collate(batch):
            bags, ys, pids, paths = zip(*batch)
            bags = torch.stack(bags, dim=0)   # [B, N, 3, H, W]
            ys = torch.stack(ys, dim=0)       # [B, L]
            return bags, ys, pids, paths

        tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate)
        va_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ Using device:", device)
        from mil_train import PatientMILMultiLabel
        model = PatientMILMultiLabel(
            n_labels=len(args.label_cols),
            encoder_name=args.encoder,
            pretrained=bool(args.pretrained),
            weights_path=args.weights_path
        ).to(device)


        pos_weight = compute_pos_weight(df_tr[args.label_cols].values.astype('float32'))
        pos_weight = pos_weight.to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_auc = -1
        best_path = os.path.join(args.out_dir, f'best_fold{fold}.ckpt')

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss = 0.0
            for bags, ys, _, _ in tr_loader:
                bags = bags.to(device)
                ys = ys.to(device)
                logits, _, _ = model(bags)
                loss = criterion(logits, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * bags.size(0)
            tr_loss /= len(tr_loader.dataset)

            # 验证
            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for bags, ys, _, _ in va_loader:
                    bags = bags.to(device)
                    ys = ys.to(device)
                    logits, _, _ = model(bags)
                    all_logits.append(logits.sigmoid().cpu().numpy())
                    all_labels.append(ys.cpu().numpy())
            all_logits = np.concatenate(all_logits, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # 逐标签 AUC
            aucs = []
            for j in range(all_labels.shape[1]):
                try:
                    auc = roc_auc_score(all_labels[:, j], all_logits[:, j])
                except ValueError:
                    auc = np.nan
                aucs.append(auc)
            mean_auc = np.nanmean(aucs)

            print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_meanAUC={mean_auc:.4f}, per_label={aucs}")
            scheduler.step()

            if mean_auc > best_auc:
                best_auc = mean_auc
                torch.save({'model': model.state_dict(), 'label_cols': args.label_cols}, best_path)
                print(f"  -> Save best to {best_path}")

    print("训练完成。每折最优权重见 outputs/ 目录。")

if __name__ == '__main__':
    main()


"""
训练
python train.py --data_root ../data --labels_csv ../labels.csv ^
  --label_cols 乙型肝炎 糖尿病 血脂异常 高尿酸血症 ^
  --epochs 20 --max_images 32 --batch_size 2 --num_workers 0 ^
  --pretrained 1 --weights_path "C:\Users\win\Desktop\crc code\weight\efficientnet_b0_rwightman-3dd342df.pth"


解释所有标签
python explain_attn_cam.py \
  --data_root data \
  --labels_csv labels.xlsx \
  --ckpt outputs/best_fold1.ckpt \
  --encoder efficientnet_b0 \
  --max_images 32 \
  --topk 5 \
  --all_labels \
  --out_dir explain_out


  python explain_attn_cam.py \
  --data_root data \
  --labels_csv labels.xlsx \
  --ckpt outputs/best_fold1.ckpt \
  --encoder efficientnet_b0 \
  --max_images 32 \
  --topk 5 \
  --target_label 0 \
  --out_dir explain_out
"""