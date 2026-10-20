"""
NeuroScan AI - Model Trainer Module
Implements:
1. Multi-Model ML Architecture (SVM, Random Forest, KNN, Decision Tree, Logistic Regression, Gradient Boosting)
2. Interactive Hyperparameter Configuration
3. Standardized Feature Scaling Pipeline
4. K-Fold Cross Validation
5. Single-Instance Diagnostic Inference with Prediction Probabilities
"""

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from .synthetic_data import DatasetManager


class ModelTrainer:
    AVAILABLE_MODELS = {
        "svm": "Support Vector Machine (SVM)",
        "random_forest": "Random Forest Classifier",
        "knn": "K-Nearest Neighbors (KNN)",
        "decision_tree": "Decision Tree Classifier",
        "logistic_regression": "Logistic Regression",
        "gradient_boosting": "Gradient Boosting Classifier"
    }

    def __init__(self, task_type: str = "binary"):
        """
        task_type: 'binary' (Healthy vs Tumor) or 'multi' (Healthy, Glioma, Meningioma, Pituitary)
        """
        self.task_type = task_type
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.trained_results: Dict[str, Any] = {}
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = DatasetManager.FEATURE_KEYS

    def _build_model_instance(self, model_name: str, params: Dict[str, Any] = None) -> Any:
        params = params or {}
        model_name = model_name.lower()

        if model_name == "svm":
            c_val = float(params.get("C", 1.0))
            kernel = params.get("kernel", "rbf")
            gamma = params.get("gamma", "scale")
            return SVC(C=c_val, kernel=kernel, gamma=gamma, probability=True, random_state=42)

        elif model_name == "random_forest":
            n_est = int(params.get("n_estimators", 100))
            max_d = int(params.get("max_depth", 10)) if params.get("max_depth") else None
            return RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)

        elif model_name == "knn":
            k = int(params.get("n_neighbors", 5))
            weights = params.get("weights", "uniform")
            return KNeighborsClassifier(n_neighbors=k, weights=weights)

        elif model_name == "decision_tree":
            criterion = params.get("criterion", "gini")
            max_d = int(params.get("max_depth", 8)) if params.get("max_depth") else None
            return DecisionTreeClassifier(criterion=criterion, max_depth=max_d, random_state=42)

        elif model_name == "logistic_regression":
            c_val = float(params.get("C", 1.0))
            return LogisticRegression(C=c_val, max_iter=1000, random_state=42)

        elif model_name == "gradient_boosting":
            n_est = int(params.get("n_estimators", 100))
            lr = float(params.get("learning_rate", 0.1))
            return GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def train_all_or_selected(
        self,
        selected_model: str = "svm",
        test_size: float = 0.20,
        hyperparams: Dict[str, Any] = None,
        train_all_benchmarks: bool = True
    ) -> Dict[str, Any]:
        """
        Train the chosen model (and optionally benchmark all algorithms).
        """
        start_time = time.time()
        X, y_multi, y_binary, feat_keys = DatasetManager.generate_feature_dataset()
        self.feature_names = feat_keys
        y = y_binary if self.task_type == "binary" else y_multi

        # Train/Test Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        self.scaler.fit(X_tr)
        self.X_train = self.scaler.transform(X_tr)
        self.X_test = self.scaler.transform(X_te)
        self.y_train = y_tr
        self.y_test = y_te

        benchmarks = {}
        models_to_train = list(self.AVAILABLE_MODELS.keys()) if train_all_benchmarks else [selected_model]

        for m_key in models_to_train:
            t0 = time.time()
            m_params = hyperparams if m_key == selected_model else {}
            clf = self._build_model_instance(m_key, m_params)
            clf.fit(self.X_train, self.y_train)
            fit_time = round((time.time() - t0) * 1000, 2) # in ms

            # Predictions
            y_pred = clf.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=5)

            self.models[m_key] = clf
            benchmarks[m_key] = {
                "name": self.AVAILABLE_MODELS[m_key],
                "accuracy": round(float(acc) * 100, 2),
                "precision": round(float(prec) * 100, 2),
                "recall": round(float(rec) * 100, 2),
                "f1_score": round(float(f1) * 100, 2),
                "cv_mean_accuracy": round(float(np.mean(cv_scores)) * 100, 2),
                "cv_std": round(float(np.std(cv_scores)) * 100, 2),
                "training_latency_ms": fit_time
            }

        self.is_trained = True
        total_time = round((time.time() - start_time) * 1000, 2)

        # Primary selected model evaluation payload
        active_clf = self.models[selected_model]
        y_pred_active = active_clf.predict(self.X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_active)

        # Calculate ROC coordinates if binary
        roc_data = None
        if self.task_type == "binary" and hasattr(active_clf, "predict_proba"):
            y_probs = active_clf.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            # Sample 20 points along curve for compact JSON transport
            step = max(1, len(fpr) // 25)
            roc_data = {
                "fpr": [round(float(x), 4) for x in fpr[::step]] + [1.0],
                "tpr": [round(float(y), 4) for y in tpr[::step]] + [1.0],
                "auc": round(float(roc_auc), 4)
            }

        self.trained_results = {
            "selected_model": selected_model,
            "selected_model_name": self.AVAILABLE_MODELS[selected_model],
            "dataset_info": {
                "total_samples": len(X),
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "num_features": len(self.feature_names),
                "test_split_ratio": f"{int(test_size * 100)}%"
            },
            "metrics": benchmarks[selected_model],
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": ["Healthy (Negative)", "Tumor (Positive)"] if self.task_type == "binary" else ["Healthy", "Glioma", "Meningioma", "Pituitary"],
                "tp": int(cm[1, 1]) if self.task_type == "binary" else 0,
                "tn": int(cm[0, 0]) if self.task_type == "binary" else 0,
                "fp": int(cm[0, 1]) if self.task_type == "binary" else 0,
                "fn": int(cm[1, 0]) if self.task_type == "binary" else 0,
            },
            "roc_curve": roc_data,
            "benchmarks": benchmarks,
            "total_execution_time_ms": total_time
        }
        return self.trained_results

    def predict_feature_vector(self, feature_vector: np.ndarray, model_name: str = "svm") -> Dict[str, Any]:
        """
        Run inference on a single feature vector extracted from an MRI image.
        """
        if not self.is_trained:
            self.train_all_or_selected(selected_model=model_name)

        model_name = model_name.lower()
        if model_name not in self.models:
            model_name = "svm"
            if "svm" not in self.models:
                self.train_all_or_selected(selected_model="svm")

        clf = self.models[model_name]
        
        # Scale the vector
        vec_reshaped = feature_vector.reshape(1, -1)
        vec_scaled = self.scaler.transform(vec_reshaped)

        pred_class = int(clf.predict(vec_scaled)[0])
        
        probabilities = {}
        confidence = 0.95

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(vec_scaled)[0]
            if self.task_type == "binary":
                probabilities = {
                    "Healthy (No Tumor)": round(float(probs[0]) * 100, 2),
                    "Brain Tumor Detected": round(float(probs[1]) * 100, 2)
                }
                confidence = round(float(probs[pred_class]) * 100, 2)
            else:
                for idx, prob in enumerate(probs):
                    label = DatasetManager.CLASS_LABELS.get(idx, f"Class {idx}")
                    probabilities[label] = round(float(prob) * 100, 2)
                confidence = round(float(probs[pred_class]) * 100, 2)
        else:
            confidence = 95.0
            probabilities = {
                "Healthy (No Tumor)": 5.0 if pred_class == 1 else 95.0,
                "Brain Tumor Detected": 95.0 if pred_class == 1 else 5.0
            }

        is_tumor = bool(pred_class != 0)
        diagnosis_title = "Brain Tumor Detected" if is_tumor else "No Tumor Detected (Healthy)"
        
        # Determine specific sub-type heuristic from GLCM features
        tumor_subtype = "None (Healthy Brain Tissue)"
        if is_tumor:
            contrast_val = feature_vector[0] if len(feature_vector) > 0 else 100
            entropy_val = feature_vector[6] if len(feature_vector) > 6 else 9.0
            if contrast_val > 95 or entropy_val > 9.0:
                tumor_subtype = "Glioma / Glioblastoma Multiforme (High Grade)"
            elif contrast_val > 70:
                tumor_subtype = "Meningioma (Dural / Extra-axial)"
            else:
                tumor_subtype = "Pituitary Adenoma (Sellar Region)"

        return {
            "prediction_class": pred_class,
            "is_tumor_detected": is_tumor,
            "diagnosis_title": diagnosis_title,
            "tumor_subtype": tumor_subtype,
            "confidence_percentage": confidence,
            "probabilities": probabilities,
            "model_used": self.AVAILABLE_MODELS.get(model_name, model_name),
            "severity_assessment": "High Severity - Urgent Neurosurgical Consultation Advised" if is_tumor else "Normal Physiological MRI Baseline"
        }
