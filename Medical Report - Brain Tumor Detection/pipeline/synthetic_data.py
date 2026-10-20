"""
NeuroScan AI - Dataset & Synthetic MRI Generator
Provides:
1. Feature Dataset Generator for ML Training (Glioma, Meningioma, Pituitary, Healthy)
2. Realistic Synthetic Brain MRI Scan Generator for testing and offline execution
"""

import numpy as np
import cv2
import os
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split


class DatasetManager:
    CLASS_LABELS = {
        0: "No Tumor (Healthy)",
        1: "Glioma Tumor",
        2: "Meningioma Tumor",
        3: "Pituitary Tumor"
    }

    BINARY_LABELS = {
        0: "Healthy (No Tumor)",
        1: "Tumor Detected"
    }

    FEATURE_KEYS = [
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation", "glcm_asm", "glcm_entropy",
        "stat_mean_intensity", "stat_std_dev", "stat_variance",
        "stat_skewness", "stat_kurtosis", "stat_rms", "stat_shannon_entropy",
        "morph_area", "morph_perimeter", "morph_circularity",
        "morph_solidity", "morph_eccentricity", "morph_aspect_ratio"
    ]

    @classmethod
    def generate_feature_dataset(cls, n_samples_per_class: int = 250, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Generate calibrated synthetic feature vectors representing clinical MRI distributions.
        Returns:
          - X (n_samples, n_features)
          - y_multi (n_samples,) [0: Healthy, 1: Glioma, 2: Meningioma, 3: Pituitary]
          - y_binary (n_samples,) [0: Healthy, 1: Tumor]
          - feature_names
        """
        np.random.seed(random_state)
        features_list = []
        labels_multi = []
        labels_binary = []

        # Class 0: Healthy / No Tumor
        # Smooth, high homogeneity, low contrast, low variance, symmetric shape
        for _ in range(n_samples_per_class):
            vec = [
                np.random.normal(25.0, 4.0),       # glcm_contrast
                np.random.normal(3.2, 0.5),        # glcm_dissimilarity
                np.random.normal(0.85, 0.04),      # glcm_homogeneity
                np.random.normal(0.32, 0.05),      # glcm_energy
                np.random.normal(0.92, 0.03),      # glcm_correlation
                np.random.normal(0.12, 0.03),      # glcm_asm
                np.random.normal(4.5, 0.5),        # glcm_entropy
                np.random.normal(85.0, 8.0),       # stat_mean_intensity
                np.random.normal(32.0, 4.0),       # stat_std_dev
                np.random.normal(1050.0, 150.0),   # stat_variance
                np.random.normal(0.05, 0.15),      # stat_skewness
                np.random.normal(0.2, 0.4),        # stat_kurtosis
                np.random.normal(90.0, 8.0),       # stat_rms
                np.random.normal(4.8, 0.3),        # stat_shannon_entropy
                np.random.normal(24000.0, 1200.0), # morph_area
                np.random.normal(680.0, 40.0),     # morph_perimeter
                np.random.normal(0.75, 0.05),      # morph_circularity
                np.random.normal(0.92, 0.03),      # morph_solidity
                np.random.normal(0.35, 0.06),      # morph_eccentricity
                np.random.normal(1.02, 0.05)       # morph_aspect_ratio
            ]
            features_list.append(vec)
            labels_multi.append(0)
            labels_binary.append(0)

        # Class 1: Glioma Tumor
        # High contrast, high entropy, high variance, irregular/infiltrative shape, low homogeneity
        for _ in range(n_samples_per_class):
            vec = [
                np.random.normal(110.0, 15.0),     # glcm_contrast
                np.random.normal(11.5, 1.8),       # glcm_dissimilarity
                np.random.normal(0.42, 0.06),      # glcm_homogeneity
                np.random.normal(0.08, 0.02),      # glcm_energy
                np.random.normal(0.68, 0.06),      # glcm_correlation
                np.random.normal(0.02, 0.01),      # glcm_asm
                np.random.normal(9.8, 0.8),        # glcm_entropy
                np.random.normal(148.0, 14.0),     # stat_mean_intensity
                np.random.normal(64.0, 8.0),       # stat_std_dev
                np.random.normal(4200.0, 500.0),   # stat_variance
                np.random.normal(1.4, 0.3),        # stat_skewness
                np.random.normal(4.5, 0.9),        # stat_kurtosis
                np.random.normal(160.0, 12.0),     # stat_rms
                np.random.normal(5.9, 0.3),        # stat_shannon_entropy
                np.random.normal(27500.0, 1800.0), # morph_area
                np.random.normal(820.0, 60.0),     # morph_perimeter
                np.random.normal(0.38, 0.06),      # morph_circularity
                np.random.normal(0.68, 0.06),      # morph_solidity
                np.random.normal(0.72, 0.08),      # morph_eccentricity
                np.random.normal(1.35, 0.15)       # morph_aspect_ratio
            ]
            features_list.append(vec)
            labels_multi.append(1)
            labels_binary.append(1)

        # Class 2: Meningioma Tumor
        # Moderate contrast, dural based, circumscribed, distinct boundary
        for _ in range(n_samples_per_class):
            vec = [
                np.random.normal(78.0, 10.0),      # glcm_contrast
                np.random.normal(7.8, 1.2),        # glcm_dissimilarity
                np.random.normal(0.55, 0.05),      # glcm_homogeneity
                np.random.normal(0.13, 0.03),      # glcm_energy
                np.random.normal(0.76, 0.05),      # glcm_correlation
                np.random.normal(0.04, 0.01),      # glcm_asm
                np.random.normal(8.1, 0.6),        # glcm_entropy
                np.random.normal(132.0, 11.0),     # stat_mean_intensity
                np.random.normal(52.0, 6.0),       # stat_std_dev
                np.random.normal(2800.0, 350.0),   # stat_variance
                np.random.normal(0.9, 0.25),       # stat_skewness
                np.random.normal(2.8, 0.6),        # stat_kurtosis
                np.random.normal(140.0, 10.0),     # stat_rms
                np.random.normal(5.5, 0.25),       # stat_shannon_entropy
                np.random.normal(25800.0, 1400.0), # morph_area
                np.random.normal(740.0, 45.0),     # morph_perimeter
                np.random.normal(0.58, 0.05),      # morph_circularity
                np.random.normal(0.82, 0.04),      # morph_solidity
                np.random.normal(0.54, 0.06),      # morph_eccentricity
                np.random.normal(1.18, 0.09)       # morph_aspect_ratio
            ]
            features_list.append(vec)
            labels_multi.append(2)
            labels_binary.append(1)

        # Class 3: Pituitary Tumor
        # Sellar region, central mass, homogeneous hyperintensity
        for _ in range(n_samples_per_class):
            vec = [
                np.random.normal(62.0, 8.0),       # glcm_contrast
                np.random.normal(6.2, 0.9),        # glcm_dissimilarity
                np.random.normal(0.62, 0.04),      # glcm_homogeneity
                np.random.normal(0.17, 0.03),      # glcm_energy
                np.random.normal(0.81, 0.04),      # glcm_correlation
                np.random.normal(0.06, 0.01),      # glcm_asm
                np.random.normal(7.3, 0.5),        # glcm_entropy
                np.random.normal(122.0, 9.0),      # stat_mean_intensity
                np.random.normal(46.0, 5.0),       # stat_std_dev
                np.random.normal(2200.0, 280.0),   # stat_variance
                np.random.normal(0.7, 0.2),        # stat_skewness
                np.random.normal(2.1, 0.5),        # stat_kurtosis
                np.random.normal(130.0, 8.0),      # stat_rms
                np.random.normal(5.2, 0.2),        # stat_shannon_entropy
                np.random.normal(25000.0, 1300.0), # morph_area
                np.random.normal(710.0, 38.0),     # morph_perimeter
                np.random.normal(0.65, 0.04),      # morph_circularity
                np.random.normal(0.86, 0.03),      # morph_solidity
                np.random.normal(0.48, 0.05),      # morph_eccentricity
                np.random.normal(1.12, 0.07)       # morph_aspect_ratio
            ]
            features_list.append(vec)
            labels_multi.append(3)
            labels_binary.append(1)

        X = np.array(features_list, dtype=np.float32)
        y_multi = np.array(labels_multi, dtype=np.int32)
        y_binary = np.array(labels_binary, dtype=np.int32)

        return X, y_multi, y_binary, cls.FEATURE_KEYS

    @classmethod
    def generate_realistic_mri_image(cls, tumor_type: str = "glioma", image_size: int = 256) -> np.ndarray:
        """
        Synthesize high-fidelity axial brain MRI simulation (parenchyma, skull, ventricles, sulci, and tumor lesion).
        tumor_type: 'none', 'glioma', 'meningioma', 'pituitary'
        """
        img = np.zeros((image_size, image_size), dtype=np.uint8)
        center = (image_size // 2, image_size // 2)

        # 1. Skull bone (high-intensity thin elliptical ring)
        cv2.ellipse(img, center, (int(image_size * 0.42), int(image_size * 0.46)), 0, 0, 360, 160, 4)
        
        # 2. Subcutaneous fat / scalp outer boundary
        cv2.ellipse(img, center, (int(image_size * 0.44), int(image_size * 0.48)), 0, 0, 360, 90, 2)

        # 3. Brain parenchyma (Cerebral hemispheres gray & white matter)
        brain_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.ellipse(brain_mask, center, (int(image_size * 0.39), int(image_size * 0.43)), 0, 0, 360, 255, -1)

        # Fill parenchyma with smooth texture
        noise = np.random.normal(75, 8, (image_size, image_size)).astype(np.uint8)
        smooth_parenchyma = cv2.GaussianBlur(noise, (15, 15), 0)
        parenchyma = cv2.bitwise_and(smooth_parenchyma, smooth_parenchyma, mask=brain_mask)
        img = cv2.add(img, parenchyma)

        # 4. Longitudinal cerebral fissure (hemisphere divider)
        cv2.line(img, (center[0], int(image_size * 0.10)), (center[0], int(image_size * 0.90)), 30, 2)

        # 5. Lateral Ventricles (CSF - dark fluid cavities)
        v_left = np.array([
            [center[0] - 12, center[1] - 40],
            [center[0] - 22, center[1] - 15],
            [center[0] - 25, center[1] + 20],
            [center[0] - 10, center[1] + 35],
            [center[0] - 8, center[1] - 10]
        ], dtype=np.int32)
        cv2.fillPoly(img, [v_left], 20)

        v_right = np.array([
            [center[0] + 12, center[1] - 40],
            [center[0] + 22, center[1] - 15],
            [center[0] + 25, center[1] + 20],
            [center[0] + 10, center[1] + 35],
            [center[0] + 8, center[1] - 10]
        ], dtype=np.int32)
        cv2.fillPoly(img, [v_right], 20)

        # 6. Brain convolutions / sulci patterns
        for angle in range(0, 360, 30):
            rad = np.deg2rad(angle)
            r1 = int(image_size * 0.28)
            r2 = int(image_size * 0.38)
            pt1 = (int(center[0] + r1 * np.cos(rad)), int(center[1] + r1 * np.sin(rad)))
            pt2 = (int(center[0] + r2 * np.cos(rad)), int(center[1] + r2 * np.sin(rad)))
            cv2.line(img, pt1, pt2, 45, 1)

        # 7. Inject specific tumor lesion if requested
        if tumor_type.lower() == "glioma":
            # Right frontal/temporal lobe, hyper-intense core with necrotic center and edematous penumbra
            t_center = (int(image_size * 0.62), int(image_size * 0.38))
            # Edema (fuzzy halo)
            cv2.circle(img, t_center, 28, 120, -1)
            # High-intensity active tumor core
            cv2.ellipse(img, t_center, (20, 16), 35, 0, 360, 215, -1)
            # Irregular micro-nodules
            cv2.circle(img, (t_center[0] + 8, t_center[1] - 6), 7, 240, -1)
            cv2.circle(img, (t_center[0] - 9, t_center[1] + 7), 5, 230, -1)
            # Central hypointense necrosis
            cv2.circle(img, t_center, 6, 60, -1)

        elif tumor_type.lower() == "meningioma":
            # Left parietal-convexity, well-demarcated homogenous mass attached to dura
            t_center = (int(image_size * 0.30), int(image_size * 0.55))
            cv2.ellipse(img, t_center, (22, 17), -25, 0, 360, 220, -1)
            # Dural tail enhancement
            cv2.line(img, (t_center[0] - 10, t_center[1] + 15), (t_center[0] - 25, t_center[1] + 30), 190, 3)

        elif tumor_type.lower() == "pituitary":
            # Sellar/suprasellar central mass expanding upward between ventricles
            t_center = (center[0], int(image_size * 0.52))
            cv2.circle(img, t_center, 15, 225, -1)
            cv2.circle(img, (t_center[0], t_center[1] - 4), 10, 245, -1)

        # Add gentle Gaussian smoothing to blend scan realistic physics
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    @classmethod
    def create_sample_gallery(cls, output_dir: str = "sample_images") -> Dict[str, str]:
        """Save a standard set of high-resolution sample MRI scans for instant UI testing."""
        os.makedirs(output_dir, exist_ok=True)
        samples = {
            "sample_glioma_case1.png": ("glioma", "Glioma Brain Tumor (Grade IV Glioblastoma)"),
            "sample_meningioma_case2.png": ("meningioma", "Meningioma Brain Tumor (Dural-based)"),
            "sample_pituitary_case3.png": ("pituitary", "Pituitary Adenoma Scan"),
            "sample_healthy_normal.png": ("none", "Normal Healthy Brain MRI Scan"),
        }
        
        saved_paths = {}
        for filename, (tumor_type, label) in samples.items():
            path = os.path.join(output_dir, filename)
            img = cls.generate_realistic_mri_image(tumor_type=tumor_type, image_size=320)
            cv2.imwrite(path, img)
            saved_paths[filename] = {
                "path": path,
                "type": tumor_type,
                "label": label
            }
        return saved_paths
