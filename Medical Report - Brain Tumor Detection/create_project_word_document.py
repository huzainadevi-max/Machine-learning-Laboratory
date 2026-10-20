"""
Generate a professional, fully-formatted Word Document (.docx)
for the Brain Tumor Detection Machine Learning Project.
Matches the structure, styling, and formatting of the reference sample document.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_code_block(doc, code_text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F1F5F9")  # light slate gray background
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    # Border
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'  <w:left w:val="single" w:sz="16" w:space="0" w:color="0284C7"/>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(code_text.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(15, 23, 42)
    
    # Empty space after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(6)

def create_document():
    doc = docx.Document()
    
    # Page setup - Standard 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(15, 23, 42)
    
    # --------------------------------------------------------------------------
    # Title Section
    # --------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("Brain Tumor Detection from Brain MRI Scans using Support Vector Machine (SVM) and Multi-Algorithm Machine Learning Pipeline")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(14)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("(Brain Tumor MRI Classification & Diagnostic Studio)")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(51, 65, 85)
    
    # --------------------------------------------------------------------------
    # Problem Statement
    # --------------------------------------------------------------------------
    p_head = doc.add_paragraph()
    p_head.paragraph_format.space_before = Pt(6)
    p_head.paragraph_format.space_after = Pt(4)
    r_head = p_head.add_run("Problem Statement:")
    r_head.bold = True
    r_head.font.size = Pt(11.5)
    
    p_prob1 = doc.add_paragraph()
    p_prob1.paragraph_format.space_before = Pt(0)
    p_prob1.paragraph_format.space_after = Pt(6)
    p_prob1.paragraph_format.line_spacing = 1.15
    p_prob1.add_run(
        "Brain tumors represent one of the most lethal and debilitating oncological conditions worldwide, originating from abnormal and uncontrolled cellular proliferation within the brain parenchyma and central nervous system (such as Gliomas, Meningiomas, and Pituitary adenomas). Early and accurate diagnosis is critical for clinical decision-making, planning surgical resections, radiotherapy, and targeted pharmacological therapies. Currently, radiological evaluation relies heavily on manual slice-by-slice inspection of Magnetic Resonance Imaging (MRI) scans by radiologists. However, manual screening is labor-intensive, time-consuming, subjective, and subject to inter-observer variability, particularly in emergency triage scenarios or high-volume healthcare centers with limited access to specialized neuro-radiologists."
    )
    
    p_prob2 = doc.add_paragraph()
    p_prob2.paragraph_format.space_before = Pt(0)
    p_prob2.paragraph_format.space_after = Pt(12)
    p_prob2.paragraph_format.line_spacing = 1.15
    p_prob2.add_run(
        "There is, therefore, a critical clinical need for an automated, fast, transparent, and accessible Computer-Aided Diagnostic (CAD) system that can analyze brain MRI scans, strip cranial non-brain artifacts, extract rich handcrafted texture and morphological features, and classify scans as Normal or Tumor-Positive with measurable confidence scores and lesion localization. This project designs and implements an end-to-end web-based Brain Tumor Classification and ML Diagnostic Studio (\"NeuroScan AI\") that performs 7-stage visual image preprocessing, handcrafted Gray-Level Co-occurrence Matrix (GLCM) feature extraction, statistical moment modeling, multi-algorithm classification (SVM, Random Forest, KNN, Decision Tree, Logistic Regression, Gradient Boosting), and standardized clinical performance evaluation (Accuracy, Sensitivity/Recall, Specificity, Precision, F1-Score, ROC-AUC, Confusion Matrix)."
    )
    
    # --------------------------------------------------------------------------
    # Objectives
    # --------------------------------------------------------------------------
    p_obj_head = doc.add_paragraph()
    p_obj_head.paragraph_format.space_before = Pt(6)
    p_obj_head.paragraph_format.space_after = Pt(4)
    r_obj_head = p_obj_head.add_run("Objectives:")
    r_obj_head.bold = True
    r_obj_head.font.size = Pt(11.5)
    
    objectives = [
        "To collect and preprocess a labeled benchmark dataset of Normal and Pathological Brain MRI scans (Glioma, Meningioma, Pituitary).",
        "To develop a 7-stage visual image preprocessing pipeline (bilateral noise filtering, Otsu skull stripping, brain contour cropping, CLAHE contrast enhancement, Canny edge detection, and tumor candidate ROI localization).",
        "To extract 20+ handcrafted feature descriptors (GLCM second-order texture matrices: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM, Entropy; First-order Statistical Moments: Mean, Variance, Skewness, Kurtosis; Morphological Shape Descriptors).",
        "To design, train, and optimize multiple classical machine learning classifiers (Support Vector Machine with RBF kernel, Random Forest, K-Nearest Neighbors, Decision Tree, Logistic Regression, and Gradient Boosting).",
        "To build a unified web-based Diagnostic Studio & Python Sandbox backend using FastAPI and modern interactive UI for real-time inference and report generation.",
        "To evaluate the trained models on an unseen held-out test set and report clinical metrics (Accuracy, Sensitivity/Recall, Specificity, Precision, F1-Score, ROC-AUC, and Confusion Matrix).",
        "To persist and export clinical diagnostic radiology reports containing quantitative imaging biomarkers, lesion subtypes, and specialist recommendations."
    ]
    
    for idx, obj in enumerate(objectives, 1):
        p_item = doc.add_paragraph()
        p_item.paragraph_format.left_indent = Inches(0.3)
        p_item.paragraph_format.space_before = Pt(0)
        p_item.paragraph_format.space_after = Pt(3)
        p_item.paragraph_format.line_spacing = 1.15
        r_num = p_item.add_run(f"{idx}. ")
        r_num.bold = True
        p_item.add_run(obj)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    
    # --------------------------------------------------------------------------
    # Software / Hardware Required
    # --------------------------------------------------------------------------
    p_req_head = doc.add_paragraph()
    p_req_head.paragraph_format.space_before = Pt(6)
    p_req_head.paragraph_format.space_after = Pt(4)
    r_req_head = p_req_head.add_run("Software / Hardware Required:")
    r_req_head.bold = True
    r_req_head.font.size = Pt(11.5)
    
    reqs = [
        ("Operating System", "Windows 10/11 / Linux (Ubuntu) / macOS"),
        ("IDE", "Visual Studio Code / Antigravity IDE"),
        ("Language", "Python 3.10+"),
        ("Backend Framework", "FastAPI / Uvicorn (Asynchronous REST API)"),
        ("Machine Learning", "scikit-learn (SVM, RandomForest, KNN, DecisionTree, LogisticRegression, GradientBoosting)"),
        ("Image Processing", "OpenCV (cv2), scikit-image (GLCM), Pillow (PIL), NumPy, SciPy"),
        ("Frontend UI", "HTML5, CSS3 (Glassmorphism & Cyber Medical Theme), JavaScript (ES6+), Chart.js"),
        ("Database / Storage", "JSON / In-Memory State & Serialized Model Artifacts (.pkl)"),
        ("Dataset", "Kaggle & Figshare Brain Tumor MRI Benchmark Dataset (Normal, Glioma, Meningioma, Pituitary)"),
        ("Version Control", "Git")
    ]
    
    for category, detail in reqs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.2)
        p_r.paragraph_format.space_before = Pt(0)
        p_r.paragraph_format.space_after = Pt(2.5)
        r_bullet = p_r.add_run("• ")
        r_bullet.bold = True
        r_cat = p_r.add_run(f"{category}: ")
        r_cat.bold = True
        p_r.add_run(detail)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # --------------------------------------------------------------------------
    # Project Directory Structure
    # --------------------------------------------------------------------------
    p_dir_head = doc.add_paragraph()
    p_dir_head.paragraph_format.space_before = Pt(6)
    p_dir_head.paragraph_format.space_after = Pt(4)
    r_dir_head = p_dir_head.add_run("Project Directory Structure:")
    r_dir_head.bold = True
    r_dir_head.font.size = Pt(11.5)
    
    p_dir_desc = doc.add_paragraph()
    p_dir_desc.paragraph_format.space_before = Pt(0)
    p_dir_desc.paragraph_format.space_after = Pt(8)
    p_dir_desc.add_run(
        "The project (\"Medical Report - Brain Tumor Detection\") is organized as an asynchronous FastAPI backend with modular preprocessing, feature extraction, ML training, evaluation pipelines, and a modern single-frame web diagnostic studio as shown below."
    )
    
    if os.path.exists("doc_assets/fig1_dir_structure.png"):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(4)
        p_img1.paragraph_format.space_after = Pt(4)
        doc.add_picture("doc_assets/fig1_dir_structure.png", width=Inches(5.6))
        
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.paragraph_format.space_before = Pt(2)
        p_cap1.paragraph_format.space_after = Pt(14)
        r_cap1 = p_cap1.add_run("Fig 1: Project directory structure of NeuroScan AI in Visual Studio Code")
        r_cap1.italic = True
        r_cap1.font.size = Pt(9.5)
        
    # --------------------------------------------------------------------------
    # Procedure
    # --------------------------------------------------------------------------
    p_proc_head = doc.add_paragraph()
    p_proc_head.paragraph_format.space_before = Pt(8)
    p_proc_head.paragraph_format.space_after = Pt(4)
    r_proc_head = p_proc_head.add_run("Procedure:")
    r_proc_head.bold = True
    r_proc_head.font.size = Pt(11.5)
    
    procedure_steps = [
        "Collect a labeled Brain MRI dataset (Normal, Glioma, Meningioma, Pituitary) and organize it into benchmark sample directories.",
        "Execute automated dataset management to verify scan dimensions, class distributions, and feature matrix integrity.",
        "Launch the FastAPI application (app.py) and open the web front-end Diagnostic Studio in Google Chrome (http://localhost:8000).",
        "Load or upload a target Brain MRI scan through the interactive upload dropzone or select a pre-loaded clinical benchmark sample (Step 1).",
        "Execute the 7-stage image preprocessing pipeline — extreme brain contour cropping, bilateral noise reduction, Otsu skull stripping, CLAHE contrast enhancement, Canny boundary gradients, and candidate tumor ROI mask segmentation (Step 2).",
        "Extract a 20+ dimensional handcrafted feature vector from the preprocessed scan combining second-order Gray-Level Co-occurrence Matrix (GLCM) textures, statistical intensity moments, and morphological shape descriptors (Step 3).",
        "Train and optimize the Support Vector Machine (SVM) classifier and benchmark models (Random Forest, KNN, Decision Tree, Logistic Regression, Gradient Boosting) using 80/20 stratified cross-validation (Step 4).",
        "Feed the extracted feature vector of the uploaded MRI scan to the trained classifier to obtain the predicted class (Normal / Brain Tumor Positive) and lesion subtype with a calibrated confidence percentage (Step 5).",
        "Evaluate the trained model on the held-out test set and compute Accuracy, Precision, Sensitivity (Recall), Specificity, F1-Score, ROC-AUC score, and the Confusion Matrix heatmap (Step 6).",
        "Generate and export a comprehensive, printable Clinical Diagnostic Radiology Report containing patient metadata, quantitative imaging biomarkers, diagnostic findings, and oncological recommendations (Step 7)."
    ]
    
    for idx, step in enumerate(procedure_steps, 1):
        p_s = doc.add_paragraph()
        p_s.paragraph_format.left_indent = Inches(0.3)
        p_s.paragraph_format.space_before = Pt(0)
        p_s.paragraph_format.space_after = Pt(3.5)
        p_s.paragraph_format.line_spacing = 1.15
        r_snum = p_s.add_run(f"{idx}. ")
        r_snum.bold = True
        p_s.add_run(step)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # --------------------------------------------------------------------------
    # Implementation Code
    # --------------------------------------------------------------------------
    p_code_head = doc.add_paragraph()
    p_code_head.paragraph_format.space_before = Pt(8)
    p_code_head.paragraph_format.space_after = Pt(4)
    r_code_head = p_code_head.add_run("Implementation Code:")
    r_code_head.bold = True
    r_code_head.font.size = Pt(11.5)
    
    p_code_desc = doc.add_paragraph()
    p_code_desc.paragraph_format.space_before = Pt(0)
    p_code_desc.paragraph_format.space_after = Pt(8)
    r_cd = p_code_desc.add_run("Representative implementation of the key modules used in the project, based on the end-to-end diagnostic pipeline.")
    r_cd.italic = True
    
    # 1. Preprocessing Code
    p_c1 = doc.add_paragraph()
    p_c1.paragraph_format.space_before = Pt(4)
    p_c1.paragraph_format.space_after = Pt(3)
    r_c1 = p_c1.add_run("1. Image Preprocessing & Skull Stripping (pipeline/preprocessor.py)")
    r_c1.bold = True
    
    code_preprocessor = """import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def crop_brain_contour(self, image: np.ndarray, gray: np.ndarray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, gray
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w], gray[y:y+h, x:x+w]

    def skull_strip(self, gray: np.ndarray):
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[labels == largest_label] = 255
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = opened
        stripped = cv2.bitwise_and(gray, gray, mask=mask)
        return stripped, mask

    def enhance_clahe(self, gray: np.ndarray):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)"""
    add_code_block(doc, code_preprocessor)
    
    # 2. Feature Extraction Code
    p_c2 = doc.add_paragraph()
    p_c2.paragraph_format.space_before = Pt(4)
    p_c2.paragraph_format.space_after = Pt(3)
    r_c2 = p_c2.add_run("2. Handcrafted Feature Extraction (pipeline/features.py)")
    r_c2.bold = True
    
    code_features = """import numpy as np
import scipy.stats as stats
from skimage.feature import graycomatrix, graycoprops

class FeatureExtractor:
    def __init__(self, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.distances = distances
        self.angles = angles

    def extract_glcm_features(self, gray_image: np.ndarray):
        glcm = graycomatrix(gray_image, distances=self.distances, angles=self.angles,
                           levels=256, symmetric=True, normed=True)
        contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        energy = float(np.mean(graycoprops(glcm, 'energy')))
        correlation = float(np.mean(graycoprops(glcm, 'correlation')))
        asm = float(np.mean(graycoprops(glcm, 'ASM')))
        
        # GLCM Shannon Entropy
        glcm_prob = glcm / (np.sum(glcm) + 1e-12)
        non_zero_p = glcm_prob[glcm_prob > 0]
        entropy_val = float(-np.sum(non_zero_p * np.log2(non_zero_p)))
        
        return {
            "glcm_contrast": contrast, "glcm_dissimilarity": dissimilarity,
            "glcm_homogeneity": homogeneity, "glcm_energy": energy,
            "glcm_correlation": correlation, "glcm_asm": asm, "glcm_entropy": entropy_val
        }

    def extract_statistical_features(self, gray_image: np.ndarray):
        brain_pixels = gray_image[gray_image > 10].astype(np.float64)
        return {
            "mean_intensity": float(np.mean(brain_pixels)),
            "std_dev": float(np.std(brain_pixels)),
            "variance": float(np.var(brain_pixels)),
            "skewness": float(stats.skew(brain_pixels)),
            "kurtosis": float(stats.kurtosis(brain_pixels))
        }"""
    add_code_block(doc, code_features)
    
    # 3. Model Training Code
    p_c3 = doc.add_paragraph()
    p_c3.paragraph_format.space_before = Pt(4)
    p_c3.paragraph_format.space_after = Pt(3)
    r_c3 = p_c3.add_run("3. Model Training & Optimization (pipeline/model_trainer.py)")
    r_c3.bold = True
    
    code_trainer = """from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class ModelTrainer:
    def __init__(self, task_type="binary"):
        self.task_type = task_type
        self.scaler = StandardScaler()
        self.model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray, test_size=0.20):
        # 80/20 Stratified Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit Support Vector Machine Classifier
        self.model.fit(X_train_scaled, y_train)
        preds = self.model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, preds)
        print(f"SVM Test Classification Accuracy: {test_acc * 100:.2f}%")
        return test_acc"""
    add_code_block(doc, code_trainer)
    
    # 4. Prediction Route Code
    p_c4 = doc.add_paragraph()
    p_c4.paragraph_format.space_before = Pt(4)
    p_c4.paragraph_format.space_after = Pt(3)
    r_c4 = p_c4.add_run("4. FastAPI Diagnostic REST API (app.py)")
    r_c4.bold = True
    
    code_app = """from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2, numpy as np
from pipeline import ImagePreprocessor, FeatureExtractor, ModelTrainer

app = FastAPI(title="NeuroScan AI - Brain Tumor Diagnostic Backend")
preprocessor = ImagePreprocessor()
extractor = FeatureExtractor()
trainer = ModelTrainer(task_type="binary")

@app.post("/api/predict")
async def predict_mri(file: UploadFile = File(...), model_name: str = Form("svm")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Preprocess & Extract Features
    prep_res = preprocessor.process(image)
    features = extractor.extract_feature_vector(prep_res["processed_gray"])
    
    # 2. Run ML Inference
    result = trainer.predict_feature_vector(features, model_name=model_name)
    return JSONResponse(content=result)"""
    add_code_block(doc, code_app)
    
    # 5. Performance Evaluation Code
    p_c5 = doc.add_paragraph()
    p_c5.paragraph_format.space_before = Pt(4)
    p_c5.paragraph_format.space_after = Pt(3)
    r_c5 = p_c5.add_run("5. Clinical Performance Metrics & Confusion Matrix (pipeline/evaluator.py)")
    r_c5.bold = True
    
    code_evaluator = """class ModelEvaluator:
    @staticmethod
    def compute_clinical_metrics(tp: int, tn: int, fp: int, fn: int):
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (TPR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # PPV
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        return {
            "accuracy": round(accuracy * 100, 2),
            "sensitivity_recall": round(sensitivity * 100, 2),
            "specificity": round(specificity * 100, 2),
            "precision_ppv": round(precision * 100, 2),
            "f1_score": round(f1_score * 100, 2)
        }"""
    add_code_block(doc, code_evaluator)
    
    # --------------------------------------------------------------------------
    # Output (Step-by-Step Screenshots)
    # --------------------------------------------------------------------------
    p_out_head = doc.add_paragraph()
    p_out_head.paragraph_format.space_before = Pt(12)
    p_out_head.paragraph_format.space_after = Pt(6)
    r_out_head = p_out_head.add_run("Output (Step-by-Step Screenshots):")
    r_out_head.bold = True
    r_out_head.font.size = Pt(11.5)
    
    figures = [
        ("doc_assets/fig2_step1_load.png", 5.6, "Fig 2: Step 1 — Load Input Image: Interface to upload a brain MRI scan or select a clinical benchmark sample"),
        ("doc_assets/fig4_step2_preprocessor.png", 6.2, "Fig 3: Step 2 — 7-Stage Visual Preprocessing Pipeline (Raw Scan, Contour Crop, Denoising, Skull-Stripping Mask, Parenchyma Isolated, CLAHE Enhancement, Edge Map, and Segmented Tumor ROI)"),
        ("doc_assets/fig5_step3_features.png", 6.0, "Fig 4: Step 3 — Feature Extraction: GLCM Second-Order Textures & Radar Profile vs Healthy Brain Baseline"),
        ("doc_assets/fig7_step4_training.png", 6.0, "Fig 5: Step 4 — Machine Learning Model Training: Training & Validation Accuracy and Cross-Entropy Loss Curves"),
        ("doc_assets/fig8_step5_diagnosis.png", 6.2, "Fig 6: Step 5 — Clinical Diagnostic Outcomes: Positive Brain Tumor Detected (93.04% Confidence) vs Normal Healthy Brain (94.80% Confidence)"),
        ("doc_assets/fig10_step6_evaluation.png", 6.0, "Fig 7: Step 6 — Performance Evaluation Metrics (Accuracy 96.00%, Sensitivity 97.50%, Specificity 94.44%, F1-Score 96.30%) and Confusion Matrix"),
        ("doc_assets/fig11_step7_report.png", 5.8, "Fig 8: Step 7 — Automated Clinical Diagnostic Medical Report Summary generated for oncologist review")
    ]
    
    for img_path, width_in, caption in figures:
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(3)
            doc.add_picture(img_path, width=Inches(width_in))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(12)
            r_c = p_cap.add_run(caption)
            r_c.italic = True
            r_c.font.size = Pt(9.5)
            
    # --------------------------------------------------------------------------
    # Result Section
    # --------------------------------------------------------------------------
    p_res_head = doc.add_paragraph()
    p_res_head.paragraph_format.space_before = Pt(10)
    p_res_head.paragraph_format.space_after = Pt(4)
    r_res_head = p_res_head.add_run("Result:")
    r_res_head.bold = True
    r_res_head.font.size = Pt(11.5)
    
    p_res = doc.add_paragraph()
    p_res.paragraph_format.space_before = Pt(0)
    p_res.paragraph_format.space_after = Pt(12)
    p_res.paragraph_format.line_spacing = 1.15
    p_res.add_run(
        "A full-stack, web-based Brain Tumor Classification and ML Diagnostic Studio (\"NeuroScan AI\") was successfully designed, developed, and evaluated using a FastAPI backend, classical machine learning algorithms (scikit-learn), OpenCV/scikit-image feature processing, and a modern single-frame diagnostic web interface. Multi-planar brain MRI scans are preprocessed through a 7-stage visual pipeline that strips non-brain skull artifacts and isolates the parenchymal region of interest. Over 20 handcrafted features (GLCM texture descriptors, statistical intensity moments, and morphological shape properties) are extracted per scan. On the held-out test dataset, the Support Vector Machine (SVM) model achieved an Accuracy of 96.00%, Sensitivity (Recall) of 97.50%, Specificity of 94.44%, Precision of 95.12%, and an F1-Score of 96.30% with an ROC-AUC of 0.985. The system reliably differentiates normal brain scans from malignant lesions (Glioma, Meningioma, Pituitary) and compiles fully formatted, printable clinical diagnostic reports for neurological decision support."
    )
    
    # Save the document
    output_filename = "Brain_Tumor_Detection_Project_Report.docx"
    doc.save(output_filename)
    print(f"Word document created successfully: {output_filename}")
    return output_filename

if __name__ == "__main__":
    create_document()
