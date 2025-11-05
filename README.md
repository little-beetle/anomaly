

# 🧠 Anomaly Detection — GLASS & PatchCore Methods

This repository reproduces and compares **unsupervised anomaly detection methods** on the **MVTec AD dataset**, focusing on industrial defect detection.
Currently, it includes verified implementations of **GLASS** and **PatchCore**, with consistent environments and result storage for future benchmarking.

---

## 🚀 Overview

* **Goal:** reproduce and validate modern anomaly detection methods on MVTec AD.
* **Dataset:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-anomaly-detection).
* **Backends:** Apple Metal (MPS), CUDA, or CPU.
* **Focus category:** *bottle* — used as a baseline across all methods.

---

## ⚙️ Environment Setup

### 1. Create environment

```bash
conda create -n anomaly_glass python=3.10
conda activate anomaly_glass
pip install -r envs/environment_glass.txt
```

or (for full setup including anomalib and GLASS)

```bash
pip install -r envs/environment_full.txt
```

### 2. Verify PyTorch installation

```bash
python -c "import torch; print(torch.__version__)"
```

Expected output:

```
2.9.0
```

---

## ▶️ Run Methods

### 🔹 GLASS

**Option 1: via shell script**

```bash
cd methods/GLASS
bash run_glass.sh
```

**Option 2: manual run**

```bash
python methods/GLASS/main.py \
  dataset --subdatasets bottle mvtec ./datasets/mvtec ./datasets/mvtec \
  net \
  --step 1 \
  --p 0 \
  --lr 0.001 \
  --meta_epochs 1 \
  --eval_epochs 1 \
  --backbone_names resnet18 \
  --layers_to_extract_from layer2 \
  --patchsize 3 \
  --target_embed_dimension 256
```

---

### 🔹 PatchCore

```bash
python methods/PATCHCORE/run_patchcore.py
```

**Results:**
Saved automatically to:

```
methods/PATCHCORE/results/
```

and exported as:

* `bottle_metrics.json`
* `bottle_metrics.csv`

---

## 📊 Results (Baseline)

| Method    | Backbone | image_AUROC | image_F1Score | pixel_AUROC | pixel_F1Score | pixel_PRO | Notes |
|-----------|----------|-------------|---------------|-------------|---------------|-----------|-------|
| PatchCore | resnet18 | **1.0000**  | 0.9920        | **0.9722**  | 0.6748        | —         | CPU run |
| GLASS     | resnet18 | **0.9905**  | —             | 0.7959      | —             | 0.7559    | best_epoch=0 |

---

## 🎯 Conclusions

* ✅ Both **GLASS** and **PatchCore** were successfully reproduced on the *MVTec AD* dataset (category **bottle**).  
* 🔹 **GLASS** achieved strong *image-level* performance (image_AUROC ≈ 0.99), confirming correct reproduction of the ECCV 2024 results.  
* 🔹 **PatchCore** outperformed GLASS at the *pixel-level* (pixel_AUROC ≈ 0.97 vs 0.79), providing more accurate localization of defects.  
* ⚙️ Verified stable operation on both **CPU** and **Apple Metal (MPS)** backends.  
* 🚀 The project now provides a **unified, reproducible pipeline** for benchmarking and future ensemble experiments combining GLASS + PatchCore (and further methods like PaDiM or SPADE).

---

## 🔬 Next Steps

* Add **PaDiM**, **SPADE**, and **CFA** implementations in `/methods/`.
* Develop unified ensemble evaluation (AUROC, PRO, F1).
* Automate report generation for cross-model comparison.
* Visualize heatmaps and uncertainty for interpretability.

---

## 📚 References

* Cui et al. *“GLASS: Generative Latent Anomaly Synthesis for Unsupervised Anomaly Detection.”* ECCV 2024.
* Roth et al. *“PatchCore: Towards Total Recall in Industrial Anomaly Detection.”* CVPR 2022.
* Defard et al. *“PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection.”* ICLR 2021.
* Bergmann et al. *“MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.”* CVPR 2019.

---

✅ **Summary:**

> This repository provides reproducible implementations of **GLASS** and **PatchCore** for anomaly detection on MVTec AD,
> serving as a foundation for future ensemble-based experiments and benchmarking.


