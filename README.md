# RetinalAI — Diabetic Retinopathy Detection
### Machine Vision Mini-Project | APTOS 2019 Dataset | EfficientNet-B4

> **🔗 Live Demo:** [retinalai.vercel.app](https://retinalai.vercel.app) ← _replace with your actual link_

---

## 📋 What is RetinalAI?

RetinalAI is an end-to-end deep learning web application that classifies **Diabetic Retinopathy (DR) severity** from retinal fundus photographs. Upload a retinal image and get an instant severity grade with a visual explanation of what the model is looking at.

Built for the Machine Vision university mini-project using the APTOS 2019 Blindness Detection dataset.

---

## 🩺 How It Works

**1. Upload** a retinal fundus photograph via the web interface.

**2. Preprocessing** — the image is cleaned and enhanced using a classical CV pipeline:
   - Ben Graham normalization (removes uneven illumination)
   - CLAHE contrast enhancement on the LAB color space
   - Non-local means denoising to preserve vessel edges

**3. Classification** — the preprocessed image is passed through an **EfficientNet-B4** deep learning model trained to predict one of 5 DR severity grades:

| Grade | Severity | Action |
|-------|----------|--------|
| 0 | No DR | Annual eye exam |
| 1 | Mild | 9–12 month follow-up |
| 2 | Moderate | Refer in 3–6 months |
| 3 | Severe | **Urgent** referral (1 month) |
| 4 | Proliferative | **EMERGENCY** referral |

**4. Explainability** — a **Grad-CAM heatmap** overlays the regions the model focused on, highlighting lesions such as:
   - 🟡 Exudates — hard deposits from leaking blood vessels
   - 🔴 Hemorrhages — bleeding within the retina
   - 🟣 Microaneurysms — earliest sign of DR

---

## 🧠 Model Architecture

```
Input (380×380 RGB)
    ↓
EfficientNet-B4 Backbone (pretrained on ImageNet)
    ↓  [1792-dim features]
Dropout(0.4) → Linear(1792→512) → ReLU → Dropout(0.3)
    ↓
Linear(512→5) → Softmax
    ↓
DR Grade (0–4)
```

**Training setup:** AdamW optimizer · CosineAnnealingLR · Label Smoothing · Mixed Precision (AMP) · Quadratic Weighted Kappa (QWK) metric

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | *Run training to populate* |
| Quadratic Weighted Kappa (QWK) | *Run training to populate* |
| Validation Loss | *Run training to populate* |

---

## 📁 Project Structure

```
blindness-ai/
├── src/
│   ├── preprocessing.py    ← Classical CV pipeline
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
├── vercel.json
└── requirements.txt
```

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

*Built for Machine Vision Mini-Project | APTOS 2019 Dataset*

**Author: Purvaj Gaonkar**
Vidyalankar Institute of Technology