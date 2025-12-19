## Update

Recent experiments report **5-fold cross-validation** results using **image-only training**:

- **AUC:** 0.6978 ± 0.0617  
- **AUPRC:** 0.7349 ± 0.0595  
- **SENS95:** 0.2802 ± 0.1342  
- **BRIER:** 0.2364 ± 0.0236  

After integrating **urine metabolomic features** via a **linear FiLM fusion layer** on top of a **frozen Attention MIL image encoder**, performance improves under the **same 5-fold cross-validation protocol**:

- **AUC:** 0.7483 ± 0.0352  
- **AUPRC:** 0.7684 ± 0.0161  
- **SENS95:** 0.2651 ± 0.1237  
- **BRIER:** 0.2362 ± 0.0374  

This multimodal formulation follows a structure similar to  
https://arxiv.org/abs/2512.03430,  
with *image-based MIL representations* and *metabolomic feature vectors* replacing the original spatial and spectral inputs, respectively.

All multimodal fusion code corresponding to these results has been updated and pushed to the **`hutu_film`** branch.
