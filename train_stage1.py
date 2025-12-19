import argparse
import numpy as np
import pandas as pd
import warnings
import os
import builtins

warnings.filterwarnings('ignore', message='Input data has no positive sample')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from libauc.losses import AUCMLoss

from dataset_patient import PatientFeatureDataset
from mil_train import PatientMILFeatures
from utils import (
    enable_jupyter_logging,
    set_seed,
    sensitivity_at_specificity,
    CombinedLoss,
    compute_pos_weight
)


def collate(batch):
    feats, ys, pids = zip(*batch)
    return torch.stack(feats), torch.stack(ys), pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--label_cols', nargs='+', required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_feats', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--instance_strategy', type=str, default='random')
    parser.add_argument('--architecture', type=str, default='attention')
    parser.add_argument('--use_combined_loss', action='store_true')
    parser.add_argument('--auc_weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default='run1')

    args = parser.parse_args()

    base_dir = os.path.join("outputs", "stage1_mil", args.run_name)
    os.makedirs(base_dir, exist_ok=True)
    enable_jupyter_logging(os.path.join(base_dir, "train.log"))

    set_seed(args.seed)

    print("===== TRAINING START =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==========================\n")

    df = pd.read_excel(args.labels_csv) if args.labels_csv.endswith(('xls', 'xlsx')) else pd.read_csv(args.labels_csv)
    y_all = df[args.label_cols].values.astype(int)
    L = y_all.shape[1]

    if L == 1:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        y_target = y_all[:, 0]
        print("Single-label task: StratifiedKFold")
    else:
        splitter = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        y_target = y_all
        print("Multi-label task: MultilabelStratifiedKFold")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_folds_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df, y_target), 1):
        print(f"\n===== Fold {fold}/{args.folds} =====")

        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        np.save(
            os.path.join(fold_dir, "train_ids.npy"),
            df.iloc[tr_idx]["id"].astype(str).to_numpy(dtype=str)
        )
        np.save(
            os.path.join(fold_dir, "val_ids.npy"),
            df.iloc[va_idx]["id"].astype(str).to_numpy(dtype=str)
        )

        train_ds = PatientFeatureDataset(
            args.feat_dir, df.iloc[tr_idx], args.label_cols, args.max_feats,
            instance_strategy=args.instance_strategy
        )
        valid_ds = PatientFeatureDataset(
            args.feat_dir, df.iloc[va_idx], args.label_cols, args.max_feats,
            instance_strategy=args.instance_strategy
        )

        tr_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate)
        va_loader = DataLoader(valid_ds, args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate)

        model = PatientMILFeatures(
            in_dim=768,
            n_labels=L,
            d_hidden_attn=128,
            dropout=0.3,
            architecture=args.architecture
        ).to(device)

        pos_weight = compute_pos_weight(
            df.iloc[tr_idx][args.label_cols].values.astype("float32")
        ).to(device)

        criterion = CombinedLoss(pos_weight, args.auc_weight) if args.use_combined_loss \
            else nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10,
            threshold=0.002, cooldown=3, min_lr=1e-6
        )

        best_metrics = {"auc": -1.0, "auprc": np.nan, "sens95": np.nan, "brier": np.nan}

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for feats, ys, _ in tr_loader:
                feats, ys = feats.to(device), ys.to(device)
                logits, _, _ = model(feats)
                loss = criterion(logits, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * feats.size(0)
            epoch_loss /= len(tr_loader.dataset)

            model.eval()
            scores, labels = [], []
            with torch.no_grad():
                for feats, ys, _ in va_loader:
                    feats = feats.to(device)
                    logits, _, _ = model(feats)
                    scores.append(logits.sigmoid().cpu().numpy())
                    labels.append(ys.numpy())

            scores = np.concatenate(scores)
            labels = np.concatenate(labels)

            try:
                auc = roc_auc_score(labels[:, 0], scores[:, 0])
                auprc = average_precision_score(labels[:, 0], scores[:, 0])
                sens95 = sensitivity_at_specificity(labels[:, 0], scores[:, 0])
                brier = brier_score_loss(labels[:, 0], scores[:, 0])
            except ValueError:
                auc, auprc, sens95, brier = np.nan, np.nan, np.nan, np.nan

            if not np.isnan(auc) and auc > best_metrics["auc"]:
                best_metrics.update(dict(auc=auc, auprc=auprc, sens95=sens95, brier=brier))
                print(f"Epoch {epoch:03d}: loss={epoch_loss:.4f} AUROC={auc:.4f} AUPRC={auprc:.4f} "
                      f"Sens@95Spec={sens95:.4f} Brier={brier:.4f} ✓ NEW BEST")
            else:
                print(f"Epoch {epoch:03d}: loss={epoch_loss:.4f} AUROC={auc:.4f} AUPRC={auprc:.4f} "
                      f"Sens@95Spec={sens95:.4f} Brier={brier:.4f}")

            scheduler.step(auc)

        print(f"Fold {fold} BEST | AUROC={best_metrics['auc']:.4f} "
              f"AUPRC={best_metrics['auprc']:.4f} "
              f"Sens@95Spec={best_metrics['sens95']:.4f} "
              f"Brier={best_metrics['brier']:.4f}")

        torch.save(model.pool.state_dict(), os.path.join(fold_dir, "mil_pool.pt"))
        all_folds_metrics.append(best_metrics)

    print("\n===== CROSS-VALIDATION SUMMARY (mean ± std) =====")
    for key in ["auc", "auprc", "sens95", "brier"]:
        vals = [m[key] for m in all_folds_metrics]
        print(f"{key.upper():<10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
