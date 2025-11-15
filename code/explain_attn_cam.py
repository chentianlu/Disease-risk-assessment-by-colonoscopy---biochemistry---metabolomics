import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader

# from mil_model import PatientMILMultiLabel
# from dataset_patient import PatientBagDataset

"""
思路：
- 仍然用 Attention 权重筛出患者的 Top-K 关键帧（帧级证据）。
- 对每个关键帧，做“标签特异”的 Grad-CAM：
  令 f(x) 为 encoder 的全局池化向量（维度 D），分类头为 Linear(D->L) 的权重 W,b。
  对标签 l，构造标量 s = <W[:,l], f(x)> + b[l]，对 s 反传到 encoder 的目标卷积层，
  获得梯度 G 与激活 A，通过 Grad-CAM 公式得到 CAM 热图。
- 这样既不依赖第三方库，又能得到每标签的帧级可视化。
"""
def collate_patient(batch):
    bags, ys, pids, paths = zip(*batch)
    import torch
    bags = torch.stack(bags, dim=0)
    ys = torch.stack(ys, dim=0)
    return bags, ys, pids, paths

def _pick_target_layer(encoder_net):
    # 优先使用最后的卷积层；不同 backbone 名称不同，这里做了兜底
    tl = getattr(encoder_net, 'conv_head', None)
    if tl is None:
        blocks = getattr(encoder_net, 'blocks', None)
        if blocks is not None and len(blocks) > 0:
            tl = blocks[-1]
    if tl is None:
        children = list(encoder_net.children())
        if len(children) > 0:
            tl = children[-1]
        else:
            tl = encoder_net
    return tl

class _GradCAMCore:
    def __init__(self, model, target_layer, label_index):
        self.model = model              # 整个 PatientMILMultiLabel 模型
        self.encoder = model.encoder.net
        self.target_layer = target_layer
        self.label_index = label_index
        self.activations = None
        self.gradients = None
        # 注册 hook 捕获 A 与 dA
        def fwd_hook(module, inp, out):
            self.activations = out.detach()          # [B, C, H', W']
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()    # [B, C, H', W']
        self.fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> np.ndarray:
        cam = cam - cam.min()
        denom = cam.max().clamp(min=1e-6)
        cam = cam / denom
        return cam.cpu().numpy()

    def __call__(self, img_tensor):
        """
        img_tensor: [1,3,H,W] 单帧
        过程：
        1) 前向过 encoder.net 得到 f(x) ∈ R^D
        2) 构造 s = <W[:,l], f(x)> + b[l]
        3) 对 s 反传，得到目标层激活 A 与梯度 G
        4) 权重 w_k = GAP(G_k)
        5) CAM = ReLU(Σ_k w_k * A_k)
        返回：归一化 cam [H', W']，与原图大小无关；外部再插值与叠加
        """
        self.model.zero_grad()
        self.encoder.zero_grad()
        # 1) 前向到 encoder 输出向量
        feats = self.encoder(img_tensor)   # [1, D]
        # 2) 构造标签特异的标量 s
        W = self.model.cls[-1].weight     # [L, D]
        b = self.model.cls[-1].bias       # [L]
        l = self.label_index
        s = F.linear(feats, W[l:l+1, :], b[l:l+1])  # [1,1]
        # 3) 反传
        self.model.zero_grad()
        self.encoder.zero_grad()
        s.backward()
        A = self.activations              # [1, C, H', W']
        G = self.gradients                # [1, C, H', W']
        # 4) GAP 权重
        weights = G.mean(dim=(2,3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * A).sum(dim=1, keepdim=False)  # [1, H', W']
        cam = F.relu(cam)[0]              # [H', W']
        return self._normalize_cam(cam)


def explain_one_label(args, target_label_idx: int, out_dir_label: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    label_cols = ckpt['label_cols']

    from mil_train import PatientMILMultiLabel
    model = PatientMILMultiLabel(
        n_labels=len(label_cols),
        encoder_name=args.encoder,
        pretrained=False,       # ✅ 解释阶段不用再加载任何预训练
        weights_path=None
    )
    model.load_state_dict(ckpt['model'], strict=True)

    model.to(device)
    model.eval()

    from dataset_patient import PatientBagDataset
    ds = PatientBagDataset(args.data_root, args.labels_csv, label_cols,
                           max_images=args.max_images, train=False)


    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_patient)


    target_layer = _pick_target_layer(model.encoder.net)

    os.makedirs(out_dir_label, exist_ok=True)

    for bags, ys, pids, paths in loader:
        bags = bags.to(device)
        logits, pooled, attn = model(bags)
        probs = logits.sigmoid().detach().cpu().numpy()[0]  # [L]
        attn = attn.detach().cpu().numpy()[0]                # [N]
        pid = pids[0]
        img_paths = paths[0]

        # 保存该患者所有标签概率
        with open(os.path.join(out_dir_label, f"{pid}_probs.txt"), 'w', encoding='utf-8') as f:
            for name, pr in zip(label_cols, probs):
                f.write(f"{name}\t{pr:.4f}\n")

        # Top-K 帧（注意力）
        top_idx = np.argsort(attn)[::-1][:args.topk]
        np.savetxt(os.path.join(out_dir_label, f"{pid}_attn_weights.txt"), attn, fmt='%.6f')

        cam_core = _GradCAMCore(model, target_layer, target_label_idx)
        try:
            for rank, i in enumerate(top_idx):
                img = (bags[0, i].unsqueeze(0))  # [1, 3, H, W]
                img_np = bags[0, i].detach().cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)

                cam = cam_core(img)  # [H', W']
                # 上采样到输入大小
                cam_t = torch.from_numpy(cam)[None, None]
                cam_up = F.interpolate(cam_t, size=img.shape[-2:], mode='bilinear', align_corners=False)[0,0].numpy()
                # 叠加与保存
                heatmap = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-6)
                heatmap_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = (img_np*255).astype(np.uint8)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                blended = cv2.addWeighted(overlay, 0.5, heatmap_color, 0.5, 0)

                base = os.path.splitext(os.path.basename(img_paths[i]))[0]
                out_name = f"{pid}_L{target_label_idx}_{label_cols[target_label_idx]}_top{rank+1}_{base}_cam.jpg"
                cv2.imwrite(os.path.join(out_dir_label, out_name), blended)
        finally:
            cam_core.remove()

    print(f"标签 [{target_label_idx}] {label_cols[target_label_idx]} 的解释导出完成：{out_dir_label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--labels_csv', type=str, default='labels.csv')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='efficientnet_b0')
    parser.add_argument('--max_images', type=int, default=32)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--target_label', type=int, default=None, help='单个标签索引；若不提供且 --all_labels 开启，则对全部标签生成解释')
    parser.add_argument('--all_labels', action='store_true', help='对 checkpoint 中的全部标签逐一生成解释')
    parser.add_argument('--out_dir', type=str, default='explain_out')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    label_cols = ckpt['label_cols']

    if args.all_labels:
        for li in range(len(label_cols)):
            sub = os.path.join(args.out_dir, f"label_{li}_{label_cols[li]}")
            explain_one_label(args, li, sub)
    else:
        assert args.target_label is not None, "未指定 target_label，或使用 --all_labels 对全部标签导出"
        sub = os.path.join(args.out_dir, f"label_{args.target_label}_{label_cols[args.target_label]}")
        explain_one_label(args, args.target_label, sub)

if __name__ == '__main__':
    main()
"""
python explain_attn_cam.py --data_root ../data --labels_csv ../labels.csv --ckpt ./outputs/best_fold1.ckpt --encoder efficientnet_b0 --max_images 32 --topk 5 --all_labels --out_dir ../explain_out
"""