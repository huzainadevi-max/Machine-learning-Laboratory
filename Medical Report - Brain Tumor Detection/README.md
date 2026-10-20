# 🧠 NeuroScan AI - Brain Tumor Detection & ML Diagnostic Studio

An End-to-End Machine Learning Web Application and Diagnostic Pipeline for Automated Brain MRI Tumor Classification.

Covers the entire ML workflow in **one unified frame website application**:
1. **Load Input Image** (DICOM / JPEG / PNG Upload & Sample MRI Library)
2. **Preprocessor** (7 Visual Stages: Denoising, Skull-Stripping via Otsu Thresholding, Morphological Contours, CLAHE Contrast Enhancement, Bounding ROI)
3. **Feature Extraction** (GLCM Second-Order Textures, Statistical Moments, and Morphological Shape Descriptors)
4. **Machine Learning Model Training** (SVM, Random Forest, KNN, Decision Tree, Logistic Regression, Gradient Boosting)
5. **Performance Evaluation** (Confusion Matrix Heatmap, Sensitivity, Specificity, ROC-AUC Curve, and Multi-Algorithm Benchmarks)
6. **Clinical Medical Diagnostic Report Generator** (Interactive Diagnostic Findings + Printable Clinical Summary)
7. **In-App Interactive Python Sandbox & Code Studio** (Type code, compile, run, and render live Matplotlib plots)

---

## 🚀 Quick Start: How to Compile and Run

### 1. Prerequisites
- **Python 3.10+** installed
- **Git** (optional)

### 2. Setup Virtual Environment & Dependencies
Open PowerShell or Terminal in the project directory:

```powershell
# Navigate to project directory
cd "c:\Users\Lenovo\OneDrive\Desktop\Medical Report - Brain Tumor Detection"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Web Application
Start the FastAPI server:

```powershell
python app.py
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 💻 Running the Command-Line Interface (CLI)

You can compile, train, evaluate, and test without a browser:

### A. Run Automated End-to-End Demo
```powershell
python run_cli.py --demo
```

### B. Diagnose a Specific MRI Scan
```powershell
python run_cli.py --image sample_images/sample_glioma_case1.png --model svm
```

### C. Train and Benchmark Algorithms from Terminal
```powershell
# Train Random Forest with 25% test split
python run_cli.py --train --model random_forest --test-split 0.25

# Train Support Vector Machine
python run_cli.py --train --model svm --test-split 0.20
```

### D. Run Automated Test Suite
```powershell
python test_pipeline.py
```

### E. Batch Train & Export Serialized Models
```powershell
python train_models.py
```

---

## 📂 Project Architecture

```
Medical Report - Brain Tumor Detection/
├── app.py                      # FastAPI Backend Server & REST Endpoints
├── run_cli.py                  # Standalone Command-Line Diagnostic Tool
├── train_models.py             # Batch ML Model Training & Weight Exporter
├── test_pipeline.py            # Comprehensive 6-Stage Pipeline Test Suite
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
├── pipeline/                   # Core Machine Learning & Medical Imaging Engine
│   ├── __init__.py             # Package Export Interface
│   ├── preprocessor.py         # Skull-Stripping, CLAHE, Denoising, ROI Segmentation
│   ├── features.py             # GLCM Texture & Statistical Feature Extractor
│   ├── synthetic_data.py       # MRI Dataset & Realistic Scan Synthesizer
│   ├── model_trainer.py        # Scikit-Learn Multi-Algorithm Training & Cross-Val
│   ├── evaluator.py            # Clinical Metrics, Confusion Matrix, ROC-AUC
│   └── code_runner.py          # Safe In-App Python Sandbox with Matplotlib Capture
├── static/                     # Web Application Assets
│   ├── index.html              # Unified Single-Page Application (One-Frame Studio)
│   ├── css/
│   │   └── style.css           # Modern Clinical Glassmorphic Stylesheet
│   └── js/
│       └── app.js              # Frontend Controller, Charts & Sandbox Runner
└── sample_images/              # High-Resolution MRI Benchmark Scans
    ├── sample_glioma_case1.png
    ├── sample_meningioma_case2.png
    ├── sample_pituitary_case3.png
    └── sample_healthy_normal.png
```

---

## 🔬 Mathematical & Algorithmic Foundations

### 1. Gray-Level Co-occurrence Matrix (GLCM)
GLCM represents the frequency of voxel intensity combinations occurring at spatial offset $(\Delta x, \Delta y)$:
- **Contrast**: $\sum_{i,j} |i - j|^2 \cdot p(i, j)$ *(Measures local intensity variance)*
- **Dissimilarity**: $\sum_{i,j} |i - j| \cdot p(i, j)$
- **Homogeneity**: $\sum_{i,j} \frac{p(i, j)}{1 + |i - j|^2}$ *(Measures structural smoothness)*
- **Energy (ASM)**: $\sum_{i,j} p(i, j)^2$ *(Measures textural uniformity)*
- **Entropy**: $-\sum_{i,j} p(i, j) \log_2(p(i, j))$ *(Measures textural chaos/disorder)*

### 2. Skull Stripping & Otsu Thresholding
Separates intracranial parenchyma from cranial bone by maximizing between-class variance:
$$\sigma_b^2(t) = \omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2$$

### 3. Clinical Diagnostic Metrics
- **Sensitivity (Recall / True Positive Rate)**: $\frac{TP}{TP + FN}$
- **Specificity (True Negative Rate)**: $\frac{TN}{TN + FP}$
- **Precision (Positive Predictive Value)**: $\frac{TP}{TP + FP}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Matthews Correlation Coefficient (MCC)**: $\frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

---

## 🛡️ License
MIT License. Developed for research and educational purposes in medical machine learning.
