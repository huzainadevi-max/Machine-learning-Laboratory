"""
NeuroScan AI - Brain Tumor Detection Web Application Backend
FastAPI server serving API endpoints, static assets, live ML inference, and python sandbox.
"""

import os
import io
import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from pipeline import (
    ImagePreprocessor,
    FeatureExtractor,
    ModelTrainer,
    ModelEvaluator,
    DatasetManager,
    execute_python_code
)

app = FastAPI(
    title="NeuroScan AI - Brain Tumor Detection Platform",
    description="Full-stack Machine Learning Medical Diagnostic Studio",
    version="2.0.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory trainer instance
trainer = ModelTrainer(task_type="binary")
trainer.train_all_or_selected(selected_model="svm", test_size=0.20)
preprocessor = ImagePreprocessor()
extractor = FeatureExtractor()

# Ensure samples exist
DatasetManager.create_sample_gallery("sample_images")


class CodeExecutionRequest(BaseModel):
    code: str


class TrainRequest(BaseModel):
    model: str = "svm"
    test_split: float = 0.20
    hyperparams: Optional[Dict[str, Any]] = None


@app.get("/api/system-status")
def get_system_status():
    return {
        "status": "online",
        "models_trained": list(trainer.models.keys()),
        "active_model": trainer.trained_results.get("selected_model", "svm"),
        "features_supported": len(DatasetManager.FEATURE_KEYS),
        "api_version": "2.0.0"
    }


@app.get("/api/samples")
def get_samples():
    """Return available sample MRI scans for instant testing."""
    sample_dir = "sample_images"
    samples_list = []
    if os.path.exists(sample_dir):
        for fname in os.listdir(sample_dir):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(sample_dir, fname)
                img = cv2.imread(full_path)
                if img is not None:
                    thumb = cv2.resize(img, (96, 96))
                    samples_list.append({
                        "filename": fname,
                        "title": fname.replace("sample_", "").replace(".png", "").replace("_", " ").title(),
                        "thumbnail": ImagePreprocessor.to_base64(thumb),
                        "path": full_path
                    })
    return {"samples": samples_list}


@app.post("/api/preprocess")
async def preprocess_image(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None)
):
    """Run 7-stage preprocessing pipeline on uploaded image or sample."""
    try:
        img_np = None
        if file is not None:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif sample_name:
            path = os.path.join("sample_images", sample_name)
            if os.path.exists(path):
                img_np = cv2.imread(path)
            else:
                img_np = DatasetManager.generate_realistic_mri_image("glioma")
        else:
            img_np = DatasetManager.generate_realistic_mri_image("glioma")

        if img_np is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        results = preprocessor.process(img_np)
        return {
            "success": True,
            "metadata": results["metadata"],
            "tumor_candidate": results["tumor_candidate"],
            "stages": results["images"]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/extract-features")
async def extract_features_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None)
):
    """Extract GLCM, statistical moments, and morphological features."""
    try:
        img_np = None
        if file is not None:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif sample_name:
            path = os.path.join("sample_images", sample_name)
            if os.path.exists(path):
                img_np = cv2.imread(path)
            else:
                img_np = DatasetManager.generate_realistic_mri_image("glioma")
        else:
            img_np = DatasetManager.generate_realistic_mri_image("glioma")

        prep_res = preprocessor.process(img_np)
        feat_vec, detailed_feat, names = extractor.extract_feature_vector(prep_res["processed_gray"])

        return {
            "success": True,
            "feature_vector": feat_vec.tolist(),
            "detailed_features": detailed_feat,
            "feature_count": len(feat_vec)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/train")
def train_model_endpoint(req: TrainRequest):
    """Train ML models and return performance metrics & confusion matrix."""
    try:
        results = trainer.train_all_or_selected(
            selected_model=req.model,
            test_size=req.test_split,
            hyperparams=req.hyperparams,
            train_all_benchmarks=True
        )
        ranked = ModelEvaluator.generate_benchmark_summary(results["benchmarks"])
        results["benchmark_rankings"] = ranked
        return {"success": True, "data": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/predict")
async def predict_pipeline_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    model: str = Form("svm")
):
    """Full End-to-End Diagnostic Pipeline execution."""
    try:
        img_np = None
        if file is not None:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif sample_name:
            path = os.path.join("sample_images", sample_name)
            if os.path.exists(path):
                img_np = cv2.imread(path)
            else:
                img_np = DatasetManager.generate_realistic_mri_image("glioma")
        else:
            img_np = DatasetManager.generate_realistic_mri_image("glioma")

        # 1. Preprocess
        prep_res = preprocessor.process(img_np)

        # 2. Feature Extraction
        feat_vec, detailed_feat, names = extractor.extract_feature_vector(prep_res["processed_gray"])

        # 3. Model Inference
        diagnosis = trainer.predict_feature_vector(feat_vec, model_name=model)

        return {
            "success": True,
            "diagnosis": diagnosis,
            "preprocessing": {
                "metadata": prep_res["metadata"],
                "tumor_candidate": prep_res["tumor_candidate"],
                "stages": prep_res["images"]
            },
            "features": {
                "vector": feat_vec.tolist(),
                "detailed": detailed_feat
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/run-code")
def run_code_endpoint(req: CodeExecutionRequest):
    """Execute Python code in isolated sandbox and return stdout, stderr, and generated plots."""
    res = execute_python_code(req.code)
    return res


@app.get("/api/code-templates")
def get_code_templates():
    """Return pre-written editable Python programs for sandbox."""
    templates = {
        "01_load_and_preprocess": {
            "title": "1. Load Image & Preprocessing (Skull Stripping & CLAHE)",
            "description": "Loads an MRI scan, applies bilateral denoise, Otsu skull stripping, and CLAHE enhancement.",
            "code": """# 1. Load Image & Preprocessing Pipeline
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load sample MRI image
img = load_sample_mri('glioma')
preprocessor = ImagePreprocessor()
result = preprocessor.process(img)

print("=== MRI PREPROCESSING SUMMARY ===")
print(f"Original Dimensions: {result['metadata']['original_dimensions']}")
print(f"Skull-Stripped Tissue Pixels: {result['metadata']['skull_stripped_pixels']}")
print(f"Mean Intensity: {result['metadata']['mean_intensity']}")
print(f"Tumor Candidate Detected: {result['tumor_candidate']['has_candidate']}")

# Plot Preprocessing Comparison
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('1. Raw MRI Scan')
axes[0].axis('off')

axes[1].imshow(result['processed_gray'], cmap='bone')
axes[1].set_title('2. Skull Stripped + CLAHE')
axes[1].axis('off')

# Morphological edges
edges = cv2.Canny(result['processed_gray'], 50, 150)
axes[2].imshow(edges, cmap='inferno')
axes[2].set_title('3. Morphological Contours')
axes[2].axis('off')

axes[3].imshow(result['processed_rgb'])
axes[3].set_title('4. Standardized ROI')
axes[3].axis('off')

plt.tight_layout()
plt.show()
"""
        },
        "02_glcm_feature_extraction": {
            "title": "2. GLCM Texture & Statistical Feature Extraction",
            "description": "Calculates Gray-Level Co-occurrence Matrix (GLCM) contrast, energy, homogeneity, entropy, and statistical moments.",
            "code": """# 2. GLCM Texture & Statistical Feature Extraction
import numpy as np
import matplotlib.pyplot as plt

# Preprocess image
img = load_sample_mri('meningioma')
preprocessor = ImagePreprocessor()
prep_res = preprocessor.process(img)

# Extract Features
extractor = FeatureExtractor()
feat_vec, detailed, names = extractor.extract_feature_vector(prep_res['processed_gray'])

print("=== EXTRACTED FEATURE VECTOR (20 DESCRIPTORS) ===")
for name, val in zip(names, feat_vec):
    print(f"  {name:<25} : {val:.4f}")

# Plot GLCM Feature Radar / Bar Chart
glcm_items = detailed['glcm_texture']['features']
keys = [f['name'] for f in glcm_items]
vals = [f['value'] for f in glcm_items]

plt.figure(figsize=(10, 4))
bars = plt.bar(keys, vals, color=['#00e5ff', '#00b0ff', '#2979ff', '#651fff', '#d500f9', '#ff1744'])
plt.title('GLCM Texture Features (Meningioma MRI)', fontsize=13, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.xticks(rotation=20, color='white')
plt.yticks(color='white')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom', color='white', fontsize=9)
plt.tight_layout()
plt.show()
"""
        },
        "03_train_ml_models": {
            "title": "3. Train & Compare Multiple ML Algorithms (SVM, RF, KNN)",
            "description": "Trains SVM, Random Forest, KNN, Decision Tree, Logistic Regression, and computes 5-Fold Cross Validation.",
            "code": """# 3. Train & Compare Multiple ML Algorithms
import numpy as np
import matplotlib.pyplot as plt

trainer = ModelTrainer(task_type="binary")
results = trainer.train_all_or_selected(selected_model="svm", test_size=0.20)

print("=== MACHINE LEARNING BENCHMARK SUMMARY ===")
print(f"{'Model':<30} {'Accuracy':<10} {'F1-Score':<10} {'Latency':<10}")
print("-" * 65)

model_names = []
accuracies = []
f1_scores = []

for k, v in results['benchmarks'].items():
    print(f"{v['name']:<30} {v['accuracy']:<9}% {v['f1_score']:<9}% {v['training_latency_ms']}ms")
    model_names.append(k.upper())
    accuracies.append(v['accuracy'])
    f1_scores.append(v['f1_score'])

# Comparison Bar Chart
x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(10, 4.5))
plt.bar(x - width/2, accuracies, width, label='Accuracy %', color='#00e5ff')
plt.bar(x + width/2, f1_scores, width, label='F1-Score %', color='#76ff03')
plt.xticks(x, model_names, color='white')
plt.yticks(color='white')
plt.ylim(80, 105)
plt.ylabel('Percentage (%)', color='white')
plt.title('Multi-Algorithm Performance Comparison', fontsize=13, color='white')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
"""
        },
        "04_evaluate_metrics": {
            "title": "4. Confusion Matrix & ROC-AUC Curve Evaluation",
            "description": "Generates diagnostic confusion matrix, sensitivity, specificity, and ROC curve.",
            "code": """# 4. Confusion Matrix & ROC-AUC Evaluation
import numpy as np
import matplotlib.pyplot as plt

trainer = ModelTrainer(task_type="binary")
results = trainer.train_all_or_selected(selected_model="svm", test_size=0.25)

cm = results['confusion_matrix']
roc = results['roc_curve']

print(f"True Positives  (TP) : {cm['tp']}")
print(f"True Negatives  (TN) : {cm['tn']}")
print(f"False Positives (FP) : {cm['fp']}")
print(f"False Negatives (FN) : {cm['fn']}")
print(f"ROC Area Under Curve (AUC): {roc['auc']}")

# Dual Plot: Confusion Matrix & ROC Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Confusion Matrix Heatmap
matrix_data = np.array(cm['matrix'])
im = ax1.imshow(matrix_data, cmap='Blues')
ax1.set_title('Confusion Matrix Heatmap', color='white')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Healthy', 'Tumor'], color='white')
ax1.set_yticklabels(['Healthy', 'Tumor'], color='white')
ax1.set_xlabel('Predicted Label', color='white')
ax1.set_ylabel('True Label', color='white')

for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(matrix_data[i, j]), ha="center", va="center", color="red" if i!=j else "black", fontsize=14, weight='bold')

# ROC Curve
ax2.plot(roc['fpr'], roc['tpr'], color='#ff007f', lw=2.5, label=f"ROC (AUC = {roc['auc']})")
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax2.set_title('Receiver Operating Characteristic (ROC)', color='white')
ax2.set_xlabel('False Positive Rate', color='white')
ax2.set_ylabel('True Positive Rate', color='white')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""
        },
        "05_full_pipeline_predict": {
            "title": "5. End-to-End Tumor Classification & Clinical Diagnosis",
            "description": "Executes complete pipeline: Image load -> Preprocess -> GLCM Features -> SVM Classification -> Diagnostic Report.",
            "code": """# 5. Full End-to-End Tumor Diagnostic Pipeline
import numpy as np

# 1. Load MRI
img = load_sample_mri('glioma')

# 2. Preprocess
preprocessor = ImagePreprocessor()
prep_res = preprocessor.process(img)

# 3. Extract Features
extractor = FeatureExtractor()
feat_vec, detailed_feat, names = extractor.extract_feature_vector(prep_res['processed_gray'])

# 4. Train/Inference
trainer = ModelTrainer(task_type="binary")
trainer.train_all_or_selected(selected_model="svm")
diagnosis = trainer.predict_feature_vector(feat_vec, model_name="svm")

print("==================================================")
print("       CLINICAL BRAIN MRI DIAGNOSTIC REPORT       ")
print("==================================================")
print(f"Status           : {diagnosis['diagnosis_title']}")
print(f"Confidence       : {diagnosis['confidence_percentage']}%")
print(f"Sub-Type         : {diagnosis['tumor_subtype']}")
print(f"Model Applied    : {diagnosis['model_used']}")
print(f"Recommendations  : {diagnosis['severity_assessment']}")
print(f"Class Probs      : {diagnosis['probabilities']}")
print("==================================================")
"""
        }
    }
    return {"templates": templates}


# Mount static frontend directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/sample_images", StaticFiles(directory="sample_images"), name="sample_images")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>NeuroScan AI - Frontend is loading...</h1>"


if __name__ == "__main__":
    print("[*] Starting NeuroScan AI Web Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
