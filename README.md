# Text-Conditioned Vision Transformer (ViT) for Multimodal Fusion  
# 基于文本条件的视觉Transformer多模态融合框架  

> A minimal framework for fusing image and text representations via a ViT backbone —referenced by *ViewDelta: Scaling Scene Change Detection through Text-Conditioning (2024)*.  
> 一个基于 ViT 主干网络的简约多模态融合框架，用于图像与文本表示的联合学习，来源于 *ViewDelta (2024)*。  

---

## Reference / 参考文献  

**ViewDelta: Scaling Scene Change Detection through Text-Conditioning**  
Subin Varghese, Joshua Gao, Vedhus Hoskere (University of Houston)  
[arXiv:2411.07612](https://arxiv.org/abs/2411.07612)  

> Introduces a ViT-based multimodal model with **text-conditioned query tokens (SQT)** and an **agentic synthetic data pipeline**.  
> 提出了一种基于 ViT 的多模态模型，利用 **文本条件查询 token (SQT)** 并构建 **agentic 合成数据流程**，实现大规模多模态学习。  

---

## Overview  
This project explores **text-conditioned multimodal fusion**, where a **Vision Transformer (ViT)** integrates:  
- **Image embeddings** from pretrained **DINOv2** (frozen)  
- **Text embeddings** from pretrained **SigLIP** (frozen)  
- **Segmentation Query Tokens (SQT)** as learnable task embeddings  

> 本项目研究基于 **文本条件的多模态融合方法**，采用 **Vision Transformer (ViT)** 主干结构，融合：  
> - 来自预训练 **DINOv2** 的图像特征（冻结）  
> - 来自预训练 **SigLIP** 的文本特征（冻结）  
> - 可学习的 **Segmentation Query Tokens (SQT)** 作为任务嵌入。  

---

## Training Challenges / 训练难点  

Training a 12-layer ViT backbone enables strong **cross-modal interaction** between text and image embeddings, but it also demands **large-scale, diverse, and high-quality data** for convergence and generalization.  
To overcome this bottleneck, the authors proposed an **Agentic AI pipeline** to automatically generate synthetic multimodal datasets.  

> 12 层 ViT 主干能够有效地建立图像与文本之间的深层融合，但其训练依赖于**海量且高质量的数据**。  
> 为解决这一瓶颈，论文提出了一种 **Agentic AI 数据生成流程**，通过自动化合成训练样本来支持大规模模型训练。  

---

## Agentic Data Generation  

A semi-automated pipeline is used to synthesize multimodal pairs:  
**LLaVA → SAM2 → LaMa (inpainting) → GPT-4o**, combining semantic discovery, mask extraction, image editing, and text prompt generation.  

> 使用半自动化 **Agentic AI 流程** 构建图文配对数据：  
> **LLaVA → SAM2 → LaMa（图像修复）→ GPT-4o**，结合语义理解、掩膜提取、生成式修改与文本提示生成，实现高质量数据扩增。  

This approach reduces manual labeling cost and improves model robustness in multimodal learning.  
> 该方法显著降低了人工标注成本，并提升了模型在多模态学习中的泛化与鲁棒性。  

---


