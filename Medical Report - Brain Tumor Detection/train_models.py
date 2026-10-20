"""
NeuroScan AI - Batch Training and Model Persistence Script
Trains ML models on the MRI Feature Dataset and saves serialized models.
"""

import os
import sys
import joblib
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline import ModelTrainer, DatasetManager, ModelEvaluator


def train_and_export_all():
    print("==================================================")
    print("  NEUROSCAN AI : BATCH MODEL TRAINING & EXPORT    ")
    print("==================================================")

    os.makedirs("models", exist_ok=True)
    trainer = ModelTrainer(task_type="binary")
    
    print("\n[*] Training all algorithms with 5-fold cross-validation...")
    results = trainer.train_all_or_selected(selected_model="svm", test_size=0.20, train_all_benchmarks=True)

    print("\n[+] Benchmark Results Summary:")
    ranked = ModelEvaluator.generate_benchmark_summary(results["benchmarks"])
    for row in ranked:
        print(f"  #{row['rank']} {row['name']:<30} Accuracy: {row['accuracy']}% | F1: {row['f1_score']}% | Latency: {row['latency_ms']}ms")

    # Export models and scaler
    print("\n[*] Exporting trained models to 'models/' directory...")
    for model_name, model_obj in trainer.models.items():
        save_path = os.path.join("models", f"{model_name}_model.pkl")
        joblib.dump(model_obj, save_path)
        print(f"  -> Saved {model_name} model to {save_path}")

    scaler_path = os.path.join("models", "standard_scaler.pkl")
    joblib.dump(trainer.scaler, scaler_path)
    print(f"  -> Saved Feature Scaler to {scaler_path}")

    print("\n[+] All models trained and exported successfully!")


if __name__ == "__main__":
    train_and_export_all()
