"""
NeuroScan AI - Brain Tumor Detection CLI Runner
Command-Line Interface to compile, execute, train, and run the pipeline from Terminal.

Usage:
  python run_cli.py --demo
  python run_cli.py --image sample_images/sample_glioma_case1.png --model svm
  python run_cli.py --train --model random_forest --test-split 0.25
"""

import argparse
import sys
import os
import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
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
    DatasetManager
)

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}======================================================================
  [*] NEUROSCAN AI : BRAIN TUMOR DETECTION & ML WORKFLOW
  Multi-Stage Diagnostic Pipeline | GLCM Textures | Classification
======================================================================{RESET}
"""
    print(banner)


def run_training_mode(model_name: str, test_split: float):
    print(f"{YELLOW}[*] Initializing Training Engine on Brain MRI Feature Dataset...{RESET}")
    trainer = ModelTrainer(task_type="binary")
    results = trainer.train_all_or_selected(selected_model=model_name, test_size=test_split)

    print(f"\n{GREEN}{BOLD}[+] Training Complete for Selected Model: {results['selected_model_name']}{RESET}")
    print(f"  * Training Samples : {results['dataset_info']['train_samples']}")
    print(f"  * Testing Samples  : {results['dataset_info']['test_samples']}")
    print(f"  * Features Used    : {results['dataset_info']['num_features']}")
    print(f"  * Execution Time   : {results['total_execution_time_ms']} ms\n")

    m = results['metrics']
    print(f"{CYAN}{BOLD}--- Primary Performance Metrics ---{RESET}")
    print(f"  * Accuracy         : {m['accuracy']}%")
    print(f"  * Precision        : {m['precision']}%")
    print(f"  * Recall           : {m['recall']}%")
    print(f"  * F1-Score         : {m['f1_score']}%")
    print(f"  * 5-Fold CV Mean   : {m['cv_mean_accuracy']}% (+/- {m['cv_std']}%)")

    print(f"\n{MAGENTA}{BOLD}--- Algorithm Benchmark Comparison ---{RESET}")
    ranked = ModelEvaluator.generate_benchmark_summary(results['benchmarks'])
    print(f"{'Rank':<5} {'Model Name':<32} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    for row in ranked:
        print(f"#{row['rank']:<4} {row['name']:<32} {row['accuracy']:<9}% {row['precision']:<9}% {row['recall']:<9}% {row['f1_score']:<9}%")

    cm = results['confusion_matrix']
    print(f"\n{YELLOW}{BOLD}--- Confusion Matrix Diagnostics ---{RESET}")
    print(f"  * True Positives (TP)  : {cm['tp']}")
    print(f"  * True Negatives (TN)  : {cm['tn']}")
    print(f"  * False Positives (FP) : {cm['fp']}")
    print(f"  * False Negatives (FN) : {cm['fn']}")


def run_inference_mode(image_path: str, model_name: str):
    print(f"{YELLOW}[*] Loading Input Brain MRI: {image_path}{RESET}")
    if not os.path.exists(image_path):
        print(f"{RED}[!] Error: File not found at '{image_path}'{RESET}")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"{RED}[!] Error: Failed to read image file.{RESET}")
        sys.exit(1)

    # Step 1: Preprocessing
    print(f"{CYAN}[1/4] Preprocessing MRI (Grayscale, Denoise, Skull-Stripping, CLAHE)...{RESET}")
    preprocessor = ImagePreprocessor()
    prep_res = preprocessor.process(img)
    print(f"  [+] Brain cropped to {prep_res['metadata']['target_dimensions']}")
    print(f"  [+] Skull-stripped pixels: {prep_res['metadata']['skull_stripped_pixels']}")

    # Step 2: Feature Extraction
    print(f"{CYAN}[2/4] Extracting GLCM & Statistical Features...{RESET}")
    extractor = FeatureExtractor()
    feat_vec, detailed_feat, names = extractor.extract_feature_vector(prep_res["processed_gray"])
    print(f"  [+] Extracted {len(feat_vec)} dimensional feature vector:")
    for group_key, group in detailed_feat.items():
        if group_key in ["glcm_texture", "statistical_intensity", "morphological_shape"]:
            print(f"    * {group['title']}:")
            for feat in group["features"][:3]:
                print(f"       - {feat['name']}: {feat['value']} ({feat['unit']})")

    # Step 3: Model Classification
    print(f"{CYAN}[3/4] Running ML Model Classification ({model_name.upper()})...{RESET}")
    trainer = ModelTrainer(task_type="binary")
    trainer.train_all_or_selected(selected_model=model_name)
    pred_res = trainer.predict_feature_vector(feat_vec, model_name=model_name)

    # Step 4: Diagnostic Summary Report
    print(f"\n{GREEN}{BOLD}======================================================================{RESET}")
    print(f"{GREEN}{BOLD}                  CLINICAL DIAGNOSTIC REPORT                         {RESET}")
    print(f"{GREEN}{BOLD}======================================================================{RESET}")
    
    if pred_res["is_tumor_detected"]:
        status_color = RED
        status_text = f"[!] {pred_res['diagnosis_title']} (POSITIVE)"
    else:
        status_color = GREEN
        status_text = f"[+] {pred_res['diagnosis_title']} (NEGATIVE / HEALTHY)"

    print(f"  Diagnostic Finding : {status_color}{BOLD}{status_text}{RESET}")
    print(f"  Confidence Level   : {BOLD}{pred_res['confidence_percentage']}%{RESET}")
    print(f"  Lesion Sub-type    : {BOLD}{pred_res['tumor_subtype']}{RESET}")
    print(f"  Clinical Action    : {pred_res['severity_assessment']}")
    print(f"  Classification Mod : {pred_res['model_used']}")
    print(f"  Probability Dist   : {pred_res['probabilities']}")
    print(f"{GREEN}{BOLD}======================================================================{RESET}\n")


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="NeuroScan AI Brain Tumor Detection CLI")
    parser.add_argument("--image", type=str, help="Path to input MRI scan image (JPG/PNG)")
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "random_forest", "knn", "decision_tree", "logistic_regression", "gradient_boosting"], help="ML Algorithm to use")
    parser.add_argument("--train", action="store_true", help="Run model training and benchmark evaluation")
    parser.add_argument("--test-split", type=float, default=0.20, help="Test split ratio (default: 0.20)")
    parser.add_argument("--demo", action="store_true", help="Run automated end-to-end demo on sample MRI scans")

    args = parser.parse_args()

    if args.demo or (not args.image and not args.train):
        print(f"{YELLOW}[*] Running Automated End-to-End Demo...{RESET}")
        sample_path = "sample_images/sample_glioma_case1.png"
        if not os.path.exists(sample_path):
            DatasetManager.create_sample_gallery()
        run_training_mode(model_name="svm", test_split=0.20)
        run_inference_mode(image_path=sample_path, model_name="svm")
    elif args.train:
        run_training_mode(model_name=args.model, test_split=args.test_split)
    elif args.image:
        run_inference_mode(image_path=args.image, model_name=args.model)


if __name__ == "__main__":
    main()
