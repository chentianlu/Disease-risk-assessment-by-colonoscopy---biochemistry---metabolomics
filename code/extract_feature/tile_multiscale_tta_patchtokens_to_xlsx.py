# -*- coding: utf-8 -*-
"""
版本B（严格语义）：
  多尺度(整图) -> 对每个尺度：tile(切块) -> TTA -> patch tokens -> tile拼接成scale长向量
  最后：多尺度融合(对所有scale长向量求均值) -> 1张图最终特征

输出：
  xlsx: patient_id, image_path, feat_0...feat_{D-1}
排序：
  patient_id(数字升序) + image_path

用法示例（Windows PowerShell / CMD）：
  python tile_multiscale_tta_patchtokens_to_xlsx.py ^
    --data_dir "..\..\data" ^
    --output_xlsx "tile_features.xlsx" ^
    --model_size vitb14 ^
    --grid_rows 2 --grid_cols 2 ^
    --scales "0.5,0.75,1.0,1.25,1.5" ^
    --weight_path "..\..\weight\dinov2_vitb14_pretrain.pth"

Linux:
  python tile_multiscale_tta_patchtokens_to_xlsx.py \
    --data_dir ../../data \
    --output_xlsx tile_multiscale_tta_patchtokens_features.xlsx \
    --model_size vitb14 \
    --grid_rows 2 --grid_cols 2 \
    --scales "0.5,0.75,1.0,1.25,1.5" \
    --weight_path ../../weight/dinov2_vitb14_pretrain.pth
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
import timm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def list_images_in_folder(folder: Path):
    """工程级：无重复、跨平台、按后缀小写过滤 + 固定排序"""
    imgs = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            imgs.append(p)
    return sorted(set(imgs), key=lambda x: str(x))


def parse_scales(scales_str: str):
    # 允许 "0.5,0.75,1.0" 这种输入
    s = scales_str.strip()
    if not s:
        return [1.0]
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in parts]


class DINOv2Backbone:
    """
    用 timm 的 DINOv2：
      - 输出 patch tokens：CLS + patch_mean 拼接 => 2C 维
      - 兼容 timm forward_features 输出为 dict 或 tensor 的情况
    """

    def __init__(self, model_size="vitb14", weight_path=None, device=None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        name_map = {
            "vits14": "vit_small_patch14_dinov2",
            "vitb14": "vit_base_patch14_dinov2",
            "vitl14": "vit_large_patch14_dinov2",
            "vitg14": "vit_giant_patch14_dinov2",
        }
        if model_size not in name_map:
            raise ValueError(f"不支持的 model_size={model_size}")
        self.timm_name = name_map[model_size]

        # num_classes=0 -> 输出 pooled 特征（通常是 CLS pooled）
        self.model = timm.create_model(self.timm_name, pretrained=False, num_classes=0)

        # base dim（例如 vitb14 通常 768）
        self.base_dim = int(self.model.num_features)

        # patch tokens 模式：CLS + patch_mean 拼接 -> 2*base_dim
        self.out_dim = self.base_dim * 2

        if weight_path:
            self._load_weights(weight_path)

        self.model.to(self.device).eval()

    def _load_weights(self, weight_path):
        state = torch.load(weight_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # 清理常见前缀
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                k = k[6:]
            elif k.startswith("backbone."):
                k = k[9:]
            elif k.startswith("module."):
                k = k[7:]
            new_state[k] = v

        msd = self.model.state_dict()
        matched = 0
        for k, v in new_state.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k] = v
                matched += 1

        self.model.load_state_dict(msd, strict=False)
        print(f"[Backbone] loaded weights, matched_keys={matched}")

    @torch.no_grad()
    def forward_patchtokens_feat(self, x: torch.Tensor) -> np.ndarray:
        """
        输出：np.ndarray shape [2C]
          - cls_token: [1, C]
          - patch_tokens: [1, N, C] -> mean => [1, C]
          - concat => [1, 2C] -> flatten => [2C]
        """
        # timm ViT 通常有 forward_features
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            # 兜底：至少能跑 model(x)
            pooled = self.model(x)  # [1, C]
            pooled = pooled.detach().float().cpu().numpy().reshape(-1)
            return np.concatenate([pooled, pooled], axis=0).astype(np.float32)

        # 情况1：dict
        if isinstance(feats, dict):
            # 有些实现会给 x_norm_clstoken / x_norm_patchtokens
            cls_tok = feats.get("x_norm_clstoken", None) or feats.get("cls_token", None)
            patch_tok = feats.get("x_norm_patchtokens", None) or feats.get("patch_tokens", None)

            if cls_tok is not None and patch_tok is not None:
                cls_vec = cls_tok  # [1, C]
                patch_mean = patch_tok.mean(dim=1)  # [1, C]
                out = torch.cat([cls_vec, patch_mean], dim=1)  # [1, 2C]
                return out.detach().float().cpu().numpy().reshape(-1).astype(np.float32)

        # 情况2：Tensor
        if torch.is_tensor(feats):
            # 可能是 [1, N, C] 或 [1, C]
            if feats.dim() == 3:
                cls_vec = feats[:, 0, :]         # [1, C]
                patch_tok = feats[:, 1:, :]      # [1, N-1, C]
                patch_mean = patch_tok.mean(dim=1)  # [1, C]
                out = torch.cat([cls_vec, patch_mean], dim=1)  # [1, 2C]
                return out.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
            elif feats.dim() == 2:
                pooled = feats  # [1, C]
                pooled = pooled.detach().float().cpu().numpy().reshape(-1)
                return np.concatenate([pooled, pooled], axis=0).astype(np.float32)

        # 兜底
        pooled = self.model(x)  # [1, C]
        pooled = pooled.detach().float().cpu().numpy().reshape(-1)
        return np.concatenate([pooled, pooled], axis=0).astype(np.float32)


class VersionBExtractor:
    """
    版本B严格融合：
      对每个尺度：
        整图缩放 -> tile切块 -> 每个tile做TTA -> 每个tile的TTA特征均值 -> tile拼接 => scale长向量
      多尺度融合：
        所有scale长向量取均值 => 最终图特征
    """

    def __init__(self, backbone: DINOv2Backbone, grid_rows=2, grid_cols=2,
                 scales=(0.5, 0.75, 1.0, 1.25, 1.5), use_tta=True):
        self.backbone = backbone
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.scales = list(scales)
        self.use_tta = bool(use_tta)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.base_transform = T.Compose([
            T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        if self.use_tta:
            self.tta_transforms = [
                self.base_transform,
                T.Compose([
                    T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomCrop(518),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]),
                T.Compose([
                    T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomHorizontalFlip(p=1.0),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]),
                T.Compose([
                    T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomVerticalFlip(p=1.0),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]),
                T.Compose([
                    T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]),
            ]
        else:
            self.tta_transforms = [self.base_transform]

        self.num_tiles = self.grid_rows * self.grid_cols
        self.expected_dim = self.num_tiles * self.backbone.out_dim

    def _resize_whole_image(self, img: Image.Image, scale: float) -> Image.Image:
        if scale == 1.0:
            return img
        w, h = img.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)

    def _iter_tiles(self, img: Image.Image):
        """不改 tile 逻辑：网格crop + 小tile跳过(<10像素)"""
        W, H = img.size
        r, c = self.grid_rows, self.grid_cols
        tile_h = H // r
        tile_w = W // c

        for i in range(r):
            for j in range(c):
                left = j * tile_w
                upper = i * tile_h
                right = min((j + 1) * tile_w, W)
                lower = min((i + 1) * tile_h, H)

                if (right - left) < 10 or (lower - upper) < 10:
                    yield None
                    continue
                yield img.crop((left, upper, right, lower))

    def _encode_one_tile_tta(self, tile: Image.Image) -> np.ndarray:
        """单尺度下：tile -> 多个TTA -> patchtokens特征 -> 平均 => 1个tile向量(2C)"""
        feats = []
        for tfm in self.tta_transforms:
            x = tfm(tile)              # [C, H, W]
            # 👇 强制兜底
            if x.shape[1] != 518 or x.shape[2] != 518:
                x = torch.nn.functional.interpolate(
                    x.unsqueeze(0),
                    size=(518, 518),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

            x = x.unsqueeze(0).to(self.backbone.device)
            f = self.backbone.forward_patchtokens_feat(x)

            feats.append(f)
        return np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32)

    def encode_image(self, image_path: str) -> np.ndarray | None:
        """版本B：先拿到每个尺度的长向量，再对尺度长向量求平均"""
        try:
            original = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[读取失败] {image_path} err={e}")
            return None

        scale_long_vectors = []

        for scale in self.scales:
            scaled = self._resize_whole_image(original, scale)

            tile_vecs = []
            ok = True
            for tile in self._iter_tiles(scaled):
                if tile is None:
                    ok = False
                    break
                tile_vecs.append(self._encode_one_tile_tta(tile))

            if not ok:
                # 这个尺度下 tile 不完整，丢弃该尺度
                continue

            long_vec = np.concatenate(tile_vecs, axis=0)  # [num_tiles*2C]
            if long_vec.shape[0] != self.expected_dim:
                continue

            scale_long_vectors.append(long_vec)

        if len(scale_long_vectors) == 0:
            return None

        final = np.mean(np.stack(scale_long_vectors, axis=0), axis=0).astype(np.float32)
        return final


def main():
    parser = argparse.ArgumentParser("VersionB: multi-scale -> tile -> TTA -> patch tokens -> fuse -> XLSX")
    parser.add_argument("--data_dir", type=str, required=True, help="数据根目录：每个子文件夹=patient_id")
    parser.add_argument("--output_xlsx", type=str, default="tile_features.xlsx", help="输出xlsx路径")
    parser.add_argument("--model_size", type=str, default="vitb14",
                        choices=["vits14", "vitb14", "vitl14", "vitg14"])
    parser.add_argument("--weight_path", type=str, default=None, help="本地权重（可选）")
    parser.add_argument("--grid_rows", type=int, default=2)
    parser.add_argument("--grid_cols", type=int, default=2)
    parser.add_argument("--scales", type=str, default="0.5,0.75,1.0,1.25,1.5", help="逗号分隔，如 0.5,0.75,1.0")
    parser.add_argument("--no_tta", action="store_true", help="禁用TTA（仍保留多尺度+tile+patchtokens）")
    parser.add_argument("--device", type=str, default=None, help="手动指定 cuda / cpu / cuda:0 等")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"data_dir 不存在: {args.data_dir}")

    scales = parse_scales(args.scales)
    use_tta = not args.no_tta

    backbone = DINOv2Backbone(
        model_size=args.model_size,
        weight_path=args.weight_path,
        device=args.device
    )
    extractor = VersionBExtractor(
        backbone=backbone,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        scales=scales,
        use_tta=use_tta
    )

    print(f"[Device] {backbone.device}")
    print(f"[Config] model={args.model_size} base_dim={backbone.base_dim} out_dim(patchtokens)={backbone.out_dim}")
    print(f"[Config] grid={args.grid_rows}x{args.grid_cols} tiles={extractor.num_tiles}")
    print(f"[Config] scales={scales}  TTA={'ON' if use_tta else 'OFF'}  expected_dim={extractor.expected_dim}")

    # patient folders：先按名称排序（但字符串排序可能不是数字排序）
    patient_folders = sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda x: str(x))

    rows = []
    expected_dim = extractor.expected_dim

    for pf in tqdm(patient_folders, desc="Patients"):
        patient_id = pf.name.strip()
        images = list_images_in_folder(pf)

        for img_path in images:
            feat = extractor.encode_image(str(img_path))
            if feat is None:
                continue
            if feat.shape[0] != expected_dim:
                # 理论上不会发生（我们已控制尺度丢弃），防御一下
                print(f"[维度不一致跳过] {img_path} feat_dim={feat.shape[0]} expected={expected_dim}")
                continue

            row = {"patient_id": patient_id, "image_path": str(img_path)}
            for k in range(expected_dim):
                row[f"feat_{k}"] = float(feat[k])
            rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("没有生成任何特征行：请检查目录结构、图片格式、模型/权重是否可用。")

    df = pd.DataFrame(rows)

    # ✅ 显式排序：patient_id 按数字升序（如果不是纯数字就退回字符串）
    try:
        df["patient_id_num"] = df["patient_id"].astype(int)
        df = df.sort_values(by=["patient_id_num", "image_path"]).drop(columns=["patient_id_num"])
    except Exception:
        df = df.sort_values(by=["patient_id", "image_path"])

    df.to_excel(args.output_xlsx, index=False)
    print(f"\n[Saved] {args.output_xlsx}")
    print(f"[Shape] {df.shape} (rows, cols)")


if __name__ == "__main__":
    main()
