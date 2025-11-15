# mil_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

# -------- Attention-MIL 聚合 --------
class AttentionMILPool(nn.Module):
    """简单可用的 Attention-MIL：对 N×D 计算权重并做加权和"""
    def __init__(self, in_dim: int, d_hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, feats: torch.Tensor):
        """
        feats: [B, N, D]
        return:
          pooled: [B, D]
          attn:   [B, N]
        """
        h = torch.tanh(self.fc1(feats))     # [B, N, d_hidden]
        logits = self.fc2(h).squeeze(-1)    # [B, N]
        attn = torch.softmax(logits, dim=1) # [B, N]
        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)  # [B, D]
        return pooled, attn

# -------- 编码器骨干，支持离线权重 --------
class EncoderBackbone(nn.Module):
    def __init__(self, name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 weights_path: str | None = None,
                 num_classes: int = 0,
                 global_pool: str = 'avg'):
        super().__init__()
        # 先不从网上下权重
        self.net = timm.create_model(
            name, pretrained=False, num_classes=num_classes, global_pool=global_pool
        )
        # D 维度
        self.out_dim = getattr(self.net, 'num_features', None)
        if self.out_dim is None:
            # 兜底：多数 timm 模型都有这个属性
            self.out_dim = self.net.get_classifier().in_features

        # 若指定离线权重，则加载
        if pretrained and weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location='cpu')
            # timm 的权重通常是和 create_model 匹配的
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"✅ Loaded offline pretrained weights: {weights_path}")
            if missing:   print(f"  (info) missing keys: {len(missing)}")
            if unexpected:print(f"  (info) unexpected keys: {len(unexpected)}")
        else:
            if pretrained:
                print("⚠️ 预训练开启但未找到 weights_path，已跳过加载；将以随机初始化训练。")
            else:
                print("ℹ️ 未使用预训练（随机初始化）。")

    def forward(self, x):
        return self.net(x)

# -------- 整体多标签 MIL 模型 --------
class PatientMILMultiLabel(nn.Module):
    def __init__(self, n_labels: int,
                 encoder_name: str = 'efficientnet_b0',
                 d_hidden_attn: int = 128,
                 dropout: float = 0.2,
                 pretrained: bool = True,
                 weights_path: str | None = None):
        super().__init__()
        self.encoder = EncoderBackbone(
            name=encoder_name,
            pretrained=pretrained,
            weights_path=weights_path,
            num_classes=0,
            global_pool='avg'
        )
        D = self.encoder.out_dim
        self.pool = AttentionMILPool(D, d_hidden_attn)
        self.cls  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(D, n_labels)
        )

    def forward(self, bags: torch.Tensor):
        """
        bags: [B, N, 3, H, W]
        return:
          logits: [B, L]
          pooled: [B, D]
          attn:   [B, N]
        """
        B, N, C, H, W = bags.shape
        x = bags.view(B * N, C, H, W)       # [B*N, 3, H, W]
        feats = self.encoder(x)              # [B*N, D]
        D = feats.shape[-1]
        feats = feats.view(B, N, D)          # [B, N, D]
        pooled, attn = self.pool(feats)      # [B, D], [B, N]
        logits = self.cls(pooled)            # [B, L]
        return logits, pooled, attn
