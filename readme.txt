## Stage 1: Attention MIL Training

After cross-validated training, we save the **trained Attention MIL head** along with the **training and validation patient IDs for each fold**, enabling consistent and leakage-free reuse in Stage 2.

outputs/
└── stage1_mil/
    └── run1/
        ├── train.log   ← timestamped log FILE
        ├── fold_1/
        │   ├── mil_pool.pt
        │   ├── train_ids.npy
        │   └── val_ids.npy
        ├── fold_2/
        ├── fold_3/
        ├── fold_4/
        └── fold_5/


## Stage 2: Multimodal Fusion Training

Stage 2 loads the pretrained **Attention MIL head** and keeps it **frozen**. Only the **linear FiLM fusion layer** (for urine metabolomic features) and the **classifier head** are trainable.


outputs/
└── stage2_film/
    └── run1/
        ├── train.log   ← timestamped log FILE
        ├── fold_1/
        │   └── model.pt
        ├── fold_2/
        ├── fold_3/
        ├── fold_4/
        └── fold_5/
