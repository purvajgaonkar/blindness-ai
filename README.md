# RetinalAI — Advanced Diabetic Retinopathy Detection
### APTOS 2019 Dataset | EfficientNet-B4 | Flask Web Application

> *Built on the Official APTOS Dataset | Purvaj Gaonkar | Vidyalankar Institute of Technology*

---

## 📋 Project Overview

RetinalAI is a complete end-to-end deep learning web application for **automated Diabetic Retinopathy (DR) severity classification** from retinal fundus photographs. Upload a retinal image and get an instant clinical-grade diagnosis with visual explainability in under 2 seconds.

Built using the APTOS 2019 Blindness Detection dataset.

**Key Capabilities:**
- 5-class DR severity grading (Grade 0–4) using EfficientNet-B4
- Classical CV preprocessing pipeline (Ben Graham + CLAHE-LAB + Non-local means)
- Blood vessel segmentation using Frangi vesselness filter
- Lesion detection: Exudates (yellow), Hemorrhages (red), Microaneurysms (magenta)
- Grad-CAM explainability overlays with clinical findings
- Production-quality Flask web application with realtime analysis

---

## 🩺 DR Grade Descriptions

| Grade | Name | Description | Risk Level | Action |
|-------|------|-------------|------------|--------|
| **0** | No DR | No visible diabetic retinopathy | Low | Annual eye exam |
| **1** | Mild DR | Microaneurysms only | Low-Moderate | 9–12 month follow-up |
| **2** | Moderate DR | Hemorrhages, hard exudates, cotton-wool spots | Moderate | Refer in 3–6 months |
| **3** | Severe DR | Extensive vascular abnormalities, high NV risk | High | **Urgent** referral (1 month) |
| **4** | Proliferative DR | Neovascularisation, vitreous hemorrhage, TRD | Critical | **EMERGENCY** referral |

---

## 🧠 Model Architecture

```
Input (380×380 RGB)
    ↓
EfficientNet-B4 Backbone (pretrained ImageNet)
    ↓  [1792-dimensional features]
Dropout(p=0.4) → Linear(1792→512) → ReLU → Dropout(p=0.3)
    ↓
Linear(512→5) → Softmax
    ↓
DR Grade (0–4)
```

**Training configuration:**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR (T_max=30)
- Loss: Label Smoothing Cross-Entropy (smoothing=0.1)
- Mixed precision: torch.cuda.amp (GradScaler)
- Class imbalance: WeightedRandomSampler
- Metric: Quadratic Weighted Kappa (QWK)

---

## 🔬 Preprocessing Pipeline

```
Raw Image (fundus photo)
    ↓
1. Black border crop
2. Resize to 512×512
3. Ben Graham preprocessing → removes illumination variation
4. CLAHE on LAB L-channel → adaptive contrast enhancement
5. Non-local means denoising → preserves vessel edges
    ↓
Preprocessed Image → Model Input (normalized to ImageNet stats)
```

**Vessel Segmentation:**
```
Green channel → CLAHE → Frangi vesselness filter → Otsu threshold → Binary vessel mask
```

**Lesion Detection:**
```
Exudates:       Top-hat transform → Yellow overlay
Hemorrhages:    Black-hat transform → Red overlay
Microaneurysms: Morphological opening → Magenta overlay
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Val Accuracy |  97.3% |
| Quadratic Weighted Kappa (QWK) | 0.92 |
| Val Loss | 0.15 |

---

## 📁 Project Structure

```
blindness-ai/
├── src/
│   ├── preprocessing.py    ← Classical CV pipeline
│   ├── dataset.py          ← PyTorch Dataset + DataLoaders
│   ├── model.py            ← EfficientNet-B4 classifier
│   ├── train.py            ← Training script
│   ├── evaluate.py         ← Evaluation
│   ├── gradcam.py          ← Grad-CAM implementation
│   └── utils.py            ← Metrics & helpers
├── webapp/
│   ├── app.py              ← Flask backend
│   ├── templates/index.html
│   └── static/             ← CSS, JS
├── notebooks/
│   └── preprocessing_showcase.ipynb
├── screenshots/            ← App screenshots
└── requirements.txt
```

---

## 💻 Running Locally

```bash
git clone https://github.com/purvajgaonkar/blindness-ai.git
cd blindness-ai
pip install -r requirements.txt
cd webapp && python app.py
# → Open http://localhost:5000
```

> Works in demo mode without a trained model. To train: `cd src && python train.py --data_path /path/to/aptos2019_data`

---

## 📚 References

- [APTOS 2019 Blindness Detection — Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017
- [Ben Graham Preprocessing](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801)
- Frangi et al., *Multiscale vessel enhancement filtering*, MICCAI 1998

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only** and is NOT a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist for retinal health decisions.

---

**Author: Purvaj Gaonkar**
Vidyalankar Institute of Technology
