# mil_train_features.py
import torch
import torch.nn as nn


class AttentionMILPool(nn.Module):
    """
    与你 mil_train.py 同款 Attention-MIL 聚合层 :contentReference[oaicite:6]{index=6}
    feats: [B, N, D] -> pooled: [B, D], attn: [B, N]
    """
    def __init__(self, in_dim: int, d_hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, feats: torch.Tensor):
        h = torch.tanh(self.fc1(feats))       # [B, N, d_hidden]
        logits = self.fc2(h).squeeze(-1)      # [B, N]
        attn = torch.softmax(logits, dim=1)   # [B, N]
        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)  # [B, D]
        return pooled, attn


class PatientMILMultiLabelFromFeatures(nn.Module):
    """
    与 mil_train.PatientMILMultiLabel 训练结构一致：AttentionMILPool + MLP分类头 :contentReference[oaicite:7]{index=7}
    只是不再包含 encoder，直接吃 instance features。
    """
    def __init__(self, in_dim: int, n_labels: int, d_hidden_attn: int = 512, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.pool = AttentionMILPool(in_dim, d_hidden_attn)

        hidden1 = 1024
        hidden2 = 512
        hidden3 = 256
        self.cls = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),

            nn.Linear(hidden3, n_labels)
        )

    def forward(self, feats: torch.Tensor):
        """
        feats: [B, N, D]
        return:
          logits: [B, L]
          pooled: [B, D]
          attn:   [B, N]
        """
        pooled, attn = self.pool(feats)
        logits = self.cls(pooled)
        return logits, pooled, attn
