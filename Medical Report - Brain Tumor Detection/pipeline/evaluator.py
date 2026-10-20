"""
NeuroScan AI - Performance Evaluation Module
Provides:
1. Medical Diagnostic Metrics (Sensitivity, Specificity, Precision, Recall, F1, MCC, ROC-AUC)
2. Detailed Confusion Matrix Diagnostics
3. Multi-Algorithm Benchmarking and Comparison Ranking
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import classification_report, matthews_corrcoef


class ModelEvaluator:
    @staticmethod
    def compute_clinical_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
        """Compute thorough medical imaging statistical metrics."""
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (TPR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # PPV
        negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # NPV
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Matthews Correlation Coefficient
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        return {
            "accuracy": round(accuracy * 100, 2),
            "sensitivity_recall": round(sensitivity * 100, 2),
            "specificity": round(specificity * 100, 2),
            "precision_ppv": round(precision * 100, 2),
            "npv": round(negative_predictive_value * 100, 2),
            "f1_score": round(f1_score * 100, 2),
            "fpr": round(false_positive_rate * 100, 2),
            "fnr": round(false_negative_rate * 100, 2),
            "mcc": round(float(mcc), 4),
            "clinical_verdict": "High Diagnostic Reliability (Exceeds Clinical 90% Sensitivity Threshold)" if sensitivity > 0.90 else "Moderate Diagnostic Reliability"
        }

    @staticmethod
    def generate_benchmark_summary(benchmarks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sort and rank models based on F1-Score & Accuracy."""
        ranked = []
        for model_id, data in benchmarks.items():
            entry = {
                "id": model_id,
                "name": data["name"],
                "accuracy": data["accuracy"],
                "precision": data["precision"],
                "recall": data["recall"],
                "f1_score": data["f1_score"],
                "cv_accuracy": data.get("cv_mean_accuracy", data["accuracy"]),
                "latency_ms": data.get("training_latency_ms", 10.0)
            }
            ranked.append(entry)

        # Sort descending by F1-Score
        ranked.sort(key=lambda x: (x["f1_score"], x["accuracy"]), reverse=True)
        for rank_idx, item in enumerate(ranked, 1):
            item["rank"] = rank_idx
            item["badge"] = "Gold Standard (Top)" if rank_idx == 1 else ("Silver" if rank_idx == 2 else "Standard")

        return ranked
