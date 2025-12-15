# tile_extract_to_xlsx.py
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
    """工程级：无重复、跨平台、按后缀小写过滤"""
    imgs = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            imgs.append(p)
    # 去重 + 固定排序
    return sorted(set(imgs), key=lambda x: str(x))


class DINOv2Encoder:
    """只负责：加载模型 + 输出单图特征（无TTA、无patch tokens拼接）"""

    def __init__(self, model_size="vitb14", weight_path=None, device=None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name_map = {
            "vits14": "vit_small_patch14_dinov2",
            "vitb14": "vit_base_patch14_dinov2",
            "vitl14": "vit_large_patch14_dinov2",
            "vitg14": "vit_giant_patch14_dinov2",
        }
        if model_size not in model_name_map:
            raise ValueError(f"不支持的 model_size={model_size}")

        self.timm_name = model_name_map[model_size]

        # 创建结构（num_classes=0 => 输出特征）
        self.model = timm.create_model(self.timm_name, pretrained=False, num_classes=0)
        self.feat_dim = int(self.model.num_features)

        # 加载权重（保持简单：你如果权重key有前缀问题，再告诉我，我再加兼容）
        if weight_path:
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            # 常见前缀清理
            new_state = {}
            for k, v in state.items():
                if k.startswith("model."):
                    k = k[6:]
                elif k.startswith("backbone."):
                    k = k[9:]
                elif k.startswith("module."):
                    k = k[7:]
                new_state[k] = v

            # 尽量加载（不强求 strict）
            msd = self.model.state_dict()
            matched = 0
            for k, v in new_state.items():
                if k in msd and msd[k].shape == v.shape:
                    msd[k] = v
                    matched += 1
            self.model.load_state_dict(msd, strict=False)
            print(f"[Encoder] loaded weights, matched_keys={matched}")

        self.model.to(self.device).eval()

        # 固定 transform：与你 tile 逻辑一致（Resize+CenterCrop+Normalize）
        self.transform = T.Compose([
            T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_pil(self, pil_img: Image.Image) -> np.ndarray:
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(x)  # [1, C]
        return feat.detach().float().cpu().numpy().reshape(-1)  # [C]


class TileEncoder:
    """
    关键：不改变你目前 tile 的处理逻辑：
      - 整图切成 r*c
      - 每个 tile 用固定 transform
      - encoder.model(tile_tensor) 得到 [C]
      - np.concatenate(tile_features) 得到 [r*c*C]
    """

    def __init__(self, encoder: DINOv2Encoder, grid_rows=2, grid_cols=2):
        self.encoder = encoder
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def encode_image(self, image_path: str) -> np.ndarray | None:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[读取失败] {image_path} err={e}")
            return None

        W, H = img.size
        r, c = self.grid_rows, self.grid_cols
        tile_h = H // r
        tile_w = W // c

        tile_features = []

        for i in range(r):
            for j in range(c):
                left = j * tile_w
                upper = i * tile_h
                right = min((j + 1) * tile_w, W)
                lower = min((i + 1) * tile_h, H)

                if (right - left) < 10 or (lower - upper) < 10:
                    continue

                tile = img.crop((left, upper, right, lower))
                feat = self.encoder.encode_pil(tile)  # [C]
                tile_features.append(feat)

        if len(tile_features) == 0:
            return None

        return np.concatenate(tile_features, axis=0)  # [r*c*C]


def main():
    parser = argparse.ArgumentParser("Tile DINOv2 feature extraction -> XLSX (no TTA)")
    parser.add_argument("--data_dir", type=str, required=True, help="数据根目录：每个子文件夹=patient_id")
    parser.add_argument("--output_xlsx", type=str, default="tile_features.xlsx", help="输出xlsx路径")
    parser.add_argument("--model_size", type=str, default="vitb14",
                        choices=["vits14", "vitb14", "vitl14", "vitg14"])
    parser.add_argument("--weight_path", type=str, default=None, help="本地权重（可选）")
    parser.add_argument("--grid_rows", type=int, default=2)
    parser.add_argument("--grid_cols", type=int, default=2)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"data_dir 不存在: {args.data_dir}")

    encoder = DINOv2Encoder(
        model_size=args.model_size,
        weight_path=args.weight_path,
    )
    tile_encoder = TileEncoder(encoder, grid_rows=args.grid_rows, grid_cols=args.grid_cols)

    # 遍历 patient folders
    patient_folders = sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda x: str(x))

    rows = []
    expected_dim = args.grid_rows * args.grid_cols * encoder.feat_dim

    for pf in tqdm(patient_folders, desc="Patients"):
        patient_id = pf.name.strip()
        images = list_images_in_folder(pf)

        for img_path in images:
            feat = tile_encoder.encode_image(str(img_path))
            if feat is None:
                continue

            # 防御：保证维度一致
            if feat.shape[0] != expected_dim:
                # 某些图被跳过了小tile会导致维度变短，这里直接跳过，保证输出矩阵整齐
                print(f"[维度不一致跳过] {img_path} feat_dim={feat.shape[0]} expected={expected_dim}")
                continue

            row = {"patient_id": patient_id, "image_path": str(img_path)}
            for k in range(expected_dim):
                row[f"feat_{k}"] = float(feat[k])
            rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("没有生成任何特征行：请检查目录结构、图片格式、模型/权重是否可用。")

    df = pd.DataFrame(rows)

    # ================== 新增：显式排序（工程级修复） ==================
    # patient_id 是字符串，转成 int 再排序，避免 1,10,2 这种问题
    df["patient_id_num"] = df["patient_id"].astype(int)

    # 先按患者 id 排序，再按 image_path 保证同一患者内稳定
    df = df.sort_values(by=["patient_id_num", "image_path"])

    # 排完序后删除临时列
    df = df.drop(columns=["patient_id_num"])
    # ==================================================================

    df.to_excel(args.output_xlsx, index=False)
    print(f"\n[Saved] {args.output_xlsx}")
    print(f"[Shape] {df.shape} (rows, cols)")



if __name__ == "__main__":
    main()


"""
python tile_extract_to_xlsx.py \
  --data_dir "../../data" \
  --output_xlsx "tile_features.xlsx" \
  --model_size vitb14 \
  --grid_rows 2 --grid_cols 2 \
  --weight_path "../../weight/dinov2_vitb14_pretrain.pth"


"""