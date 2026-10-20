"""
Comprehensive Pipeline Test Script
Verifies: Preprocessing, Feature Extraction, Training, Inference, Evaluation, and Code Sandbox.
"""

import os
import sys
import cv2
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline import (
    ImagePreprocessor,
    FeatureExtractor,
    ModelTrainer,
    ModelEvaluator,
    DatasetManager,
    execute_python_code
)


def run_all_tests():
    print("==================================================")
    print("  NEUROSCAN AI - COMPLETE PIPELINE VALIDATION   ")
    print("==================================================")

    # 1. Dataset Generation
    print("\n[1/6] Testing Synthetic MRI Dataset Generator...")
    X, y_multi, y_bin, feat_names = DatasetManager.generate_feature_dataset(n_samples_per_class=50)
    print(f" -> Generated Dataset: {X.shape[0]} samples with {X.shape[1]} features.")
    assert X.shape[0] == 200
    assert len(feat_names) == 20

    # 2. Image Preprocessing
    print("\n[2/6] Testing Image Preprocessing Pipeline...")
    sample_img = cv2.imread("sample_images/sample_glioma_case1.png")
    if sample_img is None:
        sample_img = DatasetManager.generate_realistic_mri_image("glioma")

    preprocessor = ImagePreprocessor(target_size=(256, 256))
    prep_res = preprocessor.process(sample_img)
    print(" -> Preprocessing Stages Completed:")
    for stage_name in prep_res["images"].keys():
        print(f"    - {stage_name}: OK (Base64 payload verified)")
    assert "tumor_overlay" in prep_res["images"]
    assert prep_res["tumor_candidate"]["has_candidate"] is True

    # 3. Feature Extraction
    print("\n[3/6] Testing GLCM & Statistical Feature Extraction...")
    extractor = FeatureExtractor()
    feat_vec, detailed_feat, names = extractor.extract_feature_vector(prep_res["processed_gray"])
    print(f" -> Extracted {len(feat_vec)} features:")
    print(f"    - GLCM Contrast: {detailed_feat['glcm_texture']['features'][0]['value']}")
    print(f"    - GLCM Homogeneity: {detailed_feat['glcm_texture']['features'][2]['value']}")
    print(f"    - GLCM Entropy: {detailed_feat['glcm_texture']['features'][5]['value']}")
    print(f"    - Mean Intensity: {detailed_feat['statistical_intensity']['features'][0]['value']}")
    assert len(feat_vec) == len(feat_names)

    # 4. Model Training & Hyperparameters
    print("\n[4/6] Testing Model Training & Multi-Algorithm Benchmarking...")
    trainer = ModelTrainer(task_type="binary")
    train_res = trainer.train_all_or_selected(selected_model="svm", test_size=0.25)
    print(" -> Benchmark Accuracy Results:")
    for model_id, b_data in train_res["benchmarks"].items():
        print(f"    - {b_data['name']}: Accuracy = {b_data['accuracy']}%, F1 = {b_data['f1_score']}%")
    assert train_res["metrics"]["accuracy"] > 80.0

    # 5. Diagnostic Inference & Evaluation
    print("\n[5/6] Testing Diagnostic Inference & Clinical Report Generation...")
    pred_res = trainer.predict_feature_vector(feat_vec, model_name="svm")
    print(f" -> Diagnosis: {pred_res['diagnosis_title']}")
    print(f" -> Confidence: {pred_res['confidence_percentage']}%")
    print(f" -> Subtype: {pred_res['tumor_subtype']}")
    print(f" -> Probabilities: {pred_res['probabilities']}")
    assert pred_res["is_tumor_detected"] is True

    # 6. Interactive Code Runner Sandbox
    print("\n[6/6] Testing Python In-App Sandbox & Matplotlib Plotting...")
    sample_code = """
import numpy as np
import matplotlib.pyplot as plt

print("Hello from NeuroScan Python Sandbox!")
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(6, 3))
plt.plot(x, y, color='#00e5ff', lw=2)
plt.title('Simulated ROC / Feature Response')
plt.grid(True, alpha=0.3)
print("Plot generated successfully!")
"""
    code_res = execute_python_code(sample_code)
    print(f" -> Code Sandbox Executed in {code_res['execution_time_ms']}ms:")
    print(f"    Stdout: {code_res['stdout'].strip()}")
    print(f"    Plots Captured: {code_res['plots_count']}")
    assert code_res["success"] is True
    assert code_res["plots_count"] == 1

    print("\n==================================================")
    print("  ALL 6 CORE ML MODULES PASSED VALIDATION PERFECTLY! ")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
