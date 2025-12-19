# utils.py
import os
import builtins
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from libauc.losses import AUCMLoss


# ---------- logging ----------


def enable_jupyter_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    original_print = builtins.print

    def logged_print(*args, **kwargs):
        original_print(*args, **kwargs)

        kwargs_log = dict(kwargs)
        kwargs_log.pop("file", None)

        original_print(*args, file=log_file, **kwargs_log)
        log_file.flush()

    builtins.print = logged_print

'''
def enable_jupyter_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    original_print = builtins.print

    def logged_print(*args, **kwargs):
        original_print(*args, **kwargs)
        original_print(*args, file=log_file, **kwargs)
        log_file.flush()

    builtins.print = logged_print
'''


# ---------- reproducibility ----------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- metrics ----------
def sensitivity_at_specificity(y_true, y_score, target_spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return np.nan
    return tpr[idx[-1]]


# ---------- loss ----------
class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None, auc_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.auc = AUCMLoss()
        self.auc_weight = auc_weight

    def forward(self, logits, labels):
        return (
            (1 - self.auc_weight) * self.bce(logits, labels)
            + self.auc_weight * self.auc(torch.sigmoid(logits), labels)
        )


def compute_pos_weight(labels):
    M, L = labels.shape
    weights = []
    for j in range(L):
        p = labels[:, j].sum()
        n = M - p
        weights.append((n + 1e-6) / (p + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)
