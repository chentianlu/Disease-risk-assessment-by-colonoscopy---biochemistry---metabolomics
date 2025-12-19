import argparse
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore', message='Input data has no positive sample')

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

from utils import (
    enable_jupyter_logging,
    set_seed,
    sensitivity_at_specificity,
    CombinedLoss,
    compute_pos_weight
)

from dataset_patient import PatientMultimodalDataset
from mil_train import PatientMILFiLM


def load_fold_ids(stage1_dir, fold):
    fold_dir = os.path.join(stage1_dir, f"fold_{fold}")
    train_ids = np.load(os.path.join(fold_dir, "train_ids.npy")).astype(str)
    val_ids = np.load(os.path.join(fold_dir, "val_ids.npy")).astype(str)
    return train_ids, val_ids


def filter_df_with_clinic(df, clinic_dict):
    return df[df["id"].astype(str).isin(clinic_dict.keys())].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--clinic_pt', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)

    parser.add_argument('--label_cols', nargs='+', required=True)
    parser.add_argument('--architecture', type=str, default='attention')

    parser.add_argument('--max_feats', type=int, default=16)
    parser.add_argument('--instance_strategy', type=str, default='random')
    parser.add_argument('--use_combined_loss', action='store_true')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--auc_weight', type=float, default=0)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    base_dir = os.path.join("outputs", "stage2_mil", args.run_name)
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(base_dir, f"train_{timestamp}.log")
    enable_jupyter_logging(log_path)

    print("===== STAGE-2 CONFIG =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==========================\n")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.labels_csv.endswith((".xls", ".xlsx")):
        df = pd.read_excel(args.labels_csv)
    else:
        df = pd.read_csv(args.labels_csv)

    clinic_dict = torch.load(args.clinic_pt, weights_only=False)
    clinic_dim = next(iter(clinic_dict.values())).shape[0]

    stage1_dir = os.path.join("outputs", "stage1_mil", args.run_name)

    all_folds_metrics = []

    for fold in range(1, args.folds + 1):
        print(f"\n===== Stage-2 Fold {fold}/{args.folds} =====")

        train_ids, val_ids = load_fold_ids(stage1_dir, fold)

        train_df = filter_df_with_clinic(
            df[df["id"].astype(str).isin(train_ids)],
            clinic_dict
        )
        val_df = filter_df_with_clinic(
            df[df["id"].astype(str).isin(val_ids)],
            clinic_dict
        )

        print(f"Fold {fold}: train={len(train_df)}, val={len(val_df)}")

        train_ds = PatientMultimodalDataset(
            args.feat_dir,
            train_df,
            args.label_cols,
            clinic_dict,
            max_feats=args.max_feats,
            instance_strategy=args.instance_strategy
        )
        val_ds = PatientMultimodalDataset(
            args.feat_dir,
            val_df,
            args.label_cols,
            clinic_dict,
            max_feats=args.max_feats,
            instance_strategy=args.instance_strategy
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = PatientMILFiLM(
            in_dim=768,
            clinic_dim=clinic_dim,
            n_labels=len(args.label_cols),
            architecture=args.architecture,
            freeze_mil=True
        ).to(device)

        fold_dir = os.path.join(stage1_dir, f"fold_{fold}")
        model.pool.load_state_dict(
            torch.load(
                os.path.join(fold_dir, "mil_pool.pt"),
                map_location=device,
                weights_only=True
            )
        )

        pos_weight = compute_pos_weight(
            train_df[args.label_cols].values.astype("float32")
        ).to(device)

        criterion = CombinedLoss(pos_weight, args.auc_weight)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5
        )

        best_metrics = {
            "auc": -1.0,
            "auprc": np.nan,
            "sens95": np.nan,
            "brier": np.nan
        }

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for feats, clinic, labels, _ in train_loader:
                feats = feats.to(device)
                clinic = clinic.to(device)
                labels = labels.to(device)

                logits, _, _ = model(feats, clinic)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * feats.size(0)

            epoch_loss /= len(train_loader.dataset)

            model.eval()
            scores, ys = [], []
            with torch.no_grad():
                for feats, clinic, labels, _ in val_loader:
                    logits, _, _ = model(
                        feats.to(device), clinic.to(device)
                    )
                    scores.append(torch.sigmoid(logits).cpu().numpy())
                    ys.append(labels.numpy())

            scores = np.concatenate(scores)
            ys = np.concatenate(ys)

            try:
                auc = roc_auc_score(ys[:, 0], scores[:, 0])
                auprc = average_precision_score(ys[:, 0], scores[:, 0])
                sens95 = sensitivity_at_specificity(ys[:, 0], scores[:, 0])
                brier = brier_score_loss(ys[:, 0], scores[:, 0])
            except ValueError:
                auc, auprc, sens95, brier = np.nan, np.nan, np.nan, np.nan

            if not np.isnan(auc) and auc > best_metrics["auc"]:
                best_metrics.update(
                    dict(auc=auc, auprc=auprc, sens95=sens95, brier=brier)
                )
                print(
                    f"Epoch {epoch:03d}: loss={epoch_loss:.4f} "
                    f"AUROC={auc:.4f} AUPRC={auprc:.4f} "
                    f"Sens@95Spec={sens95:.4f} Brier={brier:.4f} ✓ NEW BEST"
                )
            else:
                print(
                    f"Epoch {epoch:03d}: loss={epoch_loss:.4f} "
                    f"AUROC={auc:.4f} AUPRC={auprc:.4f} "
                    f"Sens@95Spec={sens95:.4f} Brier={brier:.4f}"
                )

            scheduler.step(auc)

        print(
            f"Fold {fold} BEST | AUROC={best_metrics['auc']:.4f} "
            f"AUPRC={best_metrics['auprc']:.4f} "
            f"Sens@95Spec={best_metrics['sens95']:.4f} "
            f"Brier={best_metrics['brier']:.4f}"
        )

        all_folds_metrics.append(best_metrics)

    print("\n===== STAGE-2 CROSS-VALIDATION SUMMARY (mean ± std) =====")
    for key in ["auc", "auprc", "sens95", "brier"]:
        vals = [m[key] for m in all_folds_metrics]
        print(f"{key.upper():<10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
