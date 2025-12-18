## Update
Recent experiments report cross-validation results using **image-only training**:

- **AUC:** 0.6978 ± 0.0617  
- **AUPRC:** 0.7349 ± 0.0595  
- **SENS95:** 0.2802 ± 0.1342  
- **BRIER:** 0.2364 ± 0.0236  

Next, we plan to **fuse colonoscopy image features with clinical variables**, treating clinical data as a vector input and integrating it into the patient-level MIL framework. This follows a similar multimodal formulation to  
https://arxiv.org/abs/2512.03430,  
where *spatial* and *spectral* features are replaced by **image features** and **clinical feature vectors** in our context.
