# Patient-Level MIL Pipeline

This project trains a patient-level MIL model using DINOv2-extracted colonoscopy image features.

See `main.ipynb` for the full workflow.

## Features
- Input: `encoder_instance_features.xlsx` (project root)
- Output: `encoder_features/` (one `.pt` file per patient)
- Each file contains a tensor of shape `[N, 768]`

## Labels
- `labels.xlsx` (project root)
- `id` matches `patient_id`

## Training
- Attention-based MIL model
- 5-fold cross-validation
- Training logs saved to the output folder

## Dependency
Requires `libauc` for `AUCMLoss`:
```bash
pip install libauc
