# explain_attn_cam.py
import argparse
import os
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_patient import PatientBagDataset
from mil_train import PatientMILMultiLabel

# ================= COLLATE FUNCTION =================
def collate_patient(batch):
    bags_list, ys, pids, paths = zip(*batch)
    ys = torch.stack(ys, dim=0) 
    
    max_n = max(bag.size(0) for bag in bags_list)
    padded_bags = []
    for bag in bags_list:
        n = bag.size(0)
        if n < max_n:
            c, h, w = bag.size(1), bag.size(2), bag.size(3)
            pad = torch.zeros(max_n - n, c, h, w)
            bag = torch.cat([bag, pad], dim=0)
        padded_bags.append(bag)
        
    bags_tensor = torch.stack(padded_bags, dim=0) 
    return bags_tensor, ys, pids, paths

# ================= 🚀终极极简省显存 GradCAM 提取器 =================
class MILGradCAM_MemoryEfficient:
    def __init__(self, model, encoder_name):
        self.model = model
        self.hook_enabled = False  # 增加开关，控制钩子是否工作
        self.activation = None
        self.gradient = None
        
        if "vit" in encoder_name or "dinov2" in encoder_name:
            self.target_layer = self.model.encoder.net.blocks[-1].norm1
            self.is_vit = True
        else:
            self.target_layer = self.model.encoder.net.conv_head
            self.is_vit = False
            
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # 只有在开关开启时，才记录数据，避开 no_grad 的干扰
        if self.hook_enabled:
            self.activation = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        if self.hook_enabled:
            self.gradient = grad_output[0].detach()

    def get_cam_for_image(self, bags, class_idx, target_idx, feats_all_no_grad):
        """精准只为 target_idx 这一张图片计算完整的反向传播和 CAM"""
        self.model.zero_grad()
        self.activation = None
        self.gradient = None
        
        B, N, C, H, W = bags.shape
        x = bags.view(B * N, C, H, W)
        
        # 关键操作：只计算目标图片的梯度，其余全用无梯度的缓存拼装
        self.hook_enabled = True  # 【开启抓取】
        f_target = self.model.encoder(x[target_idx:target_idx+1])
        
        feats_list = []
        for i in range(N):
            if i == target_idx:
                feats_list.append(f_target[0])
            else:
                feats_list.append(feats_all_no_grad[i])
        
        feats = torch.stack(feats_list, dim=0).view(B, N, -1)
        
        pooled, attn = self.model.pool(feats)
        logits = self.model.cls(pooled)
        
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        self.hook_enabled = False # 【关闭抓取】
        
        return self.activation, self.gradient

# ================= CAM 计算核心 =================
def compute_cam(activation, gradient, is_vit):
    if activation is None or gradient is None:
        raise ValueError("钩子未能捕获激活值或梯度！")
        
    if activation.dim() == 3 and is_vit:
        activation = activation[0]
        gradient = gradient[0]
    elif activation.dim() == 4 and not is_vit:
        activation = activation[0]
        gradient = gradient[0]
        
    if is_vit:
        L_full = activation.shape[0]
        H_feat = 0
        for i in range(10): 
            rem = L_full - i
            H_feat_temp = int(math.sqrt(rem))
            if H_feat_temp * H_feat_temp == rem:
                H_feat = H_feat_temp
                spatial_tokens = rem
                break
                
        a_sp = activation[-spatial_tokens:]
        g_sp = gradient[-spatial_tokens:] 
        weights = g_sp.mean(dim=0)          
        cam = (a_sp * weights).sum(dim=-1) 
        cam = F.relu(cam)
        cam = cam.reshape(H_feat, H_feat)  
    else:
        weights = gradient.mean(dim=(1, 2), keepdim=True) 
        cam = (activation * weights).sum(dim=0)          
        cam = F.relu(cam)
        
    cam = cam - cam.min()
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
        
    return cam.cpu().numpy()

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='vit_base_patch14_dinov2')
    parser.add_argument('--max_images', type=int, default=16)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--label_index', type=int, default=0)
    parser.add_argument('--only_positive', action='store_true')
    parser.add_argument('--out_dir', type=str, default='./explain_out')
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_max_images = 0 if args.max_images == -1 else args.max_images
    df = pd.read_excel(args.labels_csv) if args.labels_csv.endswith(('xls', 'xlsx')) else pd.read_csv(args.labels_csv)
    label_cols_dummy = [df.columns[1]] 
    
    dataset = PatientBagDataset(
        root_dir=args.data_root, 
        labels_data=df, 
        label_cols=label_cols_dummy, 
        max_images=dataset_max_images, 
        train=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_patient)

    print(f"Loading checkpoint from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    label_cols = checkpoint.get('label_cols', label_cols_dummy)
    
    model = PatientMILMultiLabel(
        n_labels=len(label_cols),
        encoder_name=args.encoder,
        pretrained=False,
        freeze_encoder=False 
    )
    
    state_dict = checkpoint['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval() 

    cam_extractor = MILGradCAM_MemoryEfficient(model, args.encoder)
    
    print(f"===== 开始可解释性分析 (Target Label Index: {args.label_index}) =====")
    
    for bags, ys, pids, paths in tqdm(loader, desc="Processing Patients"):
        pid = pids[0]
        bag_paths = paths[0]
        bags = bags.to(device)
        B, N, C, H, W = bags.shape
        x = bags.view(B * N, C, H, W)
        
        # 【阶段 1】：无梯度快速摸底
        with torch.no_grad():
            feats_all_no_grad = model.encoder(x)
            feats_view = feats_all_no_grad.view(B, N, -1)
            pooled_val, attn_val = model.pool(feats_view)
            logits_val = model.cls(pooled_val)
            prob = torch.sigmoid(logits_val)[0, args.label_index].item()
            
        if args.only_positive and prob < 0.5:
            continue
            
        attn_scores = attn_val[0].cpu().numpy()
        N_valid = len(bag_paths)
        actual_k = min(args.topk, N_valid)
        topk_indices = np.argsort(attn_scores)[-actual_k:][::-1] 
        
        # 【阶段 2】：对挑选出来的高分图片，逐张提取热力图
        for rank, idx in enumerate(topk_indices):
            idx = int(idx) # 🌟 强制转换为原生 Python int，彻底掐断 KeyError 风险！
            img_path = bag_paths[idx]
            img_attn = attn_scores[idx]
            
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            H_orig, W_orig, _ = orig_img.shape
            
            # 拿到单张图的神圣抓取结果
            act, grad = cam_extractor.get_cam_for_image(bags, args.label_index, idx, feats_all_no_grad)
            cam = compute_cam(act, grad, cam_extractor.is_vit)
            cam_resized = cv2.resize(cam, (W_orig, H_orig))
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.float32(heatmap) / 255 * 0.5 + np.float32(orig_img) / 255 * 0.5
            overlay = np.uint8(255 * overlay / np.max(overlay))
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            axes[1].imshow(overlay)
            axes[1].set_title(f"GradCAM (Attn: {img_attn:.4f})")
            axes[1].axis('off')
            
            plt.suptitle(f"PID: {pid} | Pred Score: {prob:.4f} | Rank: {rank+1}")
            plt.tight_layout()
            
            save_name = f"{pid}_pred{prob:.3f}_rank{rank+1}_attn{img_attn:.3f}.png"
            plt.savefig(os.path.join(args.out_dir, save_name), dpi=150)
            plt.close(fig)

if __name__ == "__main__":
    main()