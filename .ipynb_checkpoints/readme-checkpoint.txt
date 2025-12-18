# README（请先阅读）
# =============================
# 目标：
# 1) 以患者为单位做多标签分类（如：是否糖尿病、是否疾病B…）。
# 2) 每位患者约30张肠镜图像；患者级标签在 labels.csv 中。
# 3) 使用 Attention-MIL 将一组图像聚合成患者表示，做多标签二分类。
# 4) 训练后导出：
#    - 患者级预测结果与每张图的注意力权重（可找出关键帧）。
#    - 对关键帧跑 Grad-CAM 热图，直观看模型关注区域。
#    - 可选：TCAV 概念级解释（需要你自建概念图库）。
#
# 环境建议：
#   Python ≥3.9, torch ≥2.0, torchvision ≥0.15, timm, pandas, scikit-learn, opencv-python, Pillow, pytorch-grad-cam
#   pip install timm pandas scikit-learn opencv-python pillow pytorch-grad-cam
#
# 目录组织（示例）：
#   your_project/
#     data/
#       patient_0001/
#         img_001.jpg
#         img_002.jpg
#         ...
#       patient_0002/
#         ...
#     labels.csv    # 列：patient_id, diabetes, disease_B, disease_C ... （0/1）
#     concepts/     # (可选) TCAV 概念图库，每个子文件夹是一个概念
#       redness/
#         *.jpg
#       edema/
#         *.jpg
#     code/
#       (本文件保存为多个 .py)
#
# 使用顺序：
#   1) 修改 config.yaml（或直接改 train.py 的参数区）指向你的 data 与 labels.csv。
#   2) 运行 train.py 进行训练与验证；保存 best.ckpt。
#   3) 运行 explain_attn_cam.py 生成：
#        - 每位患者的 Top-K 关键帧（按注意力权重）
#        - 关键帧的 Grad-CAM 热图
#   4) （可选）运行 tcav_probe.py 做概念级解释并输出每个概念的影响方向和显著性。
#
# 说明：
#   - 多标签损失使用 BCEWithLogitsLoss；类别不均衡时可设置 pos_weight。
#   - Attention-MIL 的注意力权重即“帧重要性”。
#   - Grad-CAM 对单帧进行；先用注意力筛出关键帧再做。
#   - TCAV 需要你提供少量“概念图片”（正例）与“对照图片”（反例），脚本会在 encoder 特征上训练线性分隔向量并计算方向导数分数。
#
# =============================