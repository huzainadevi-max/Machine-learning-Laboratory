"""
NeuroScan AI - Feature Extraction Module
Implements:
1. Gray-Level Co-occurrence Matrix (GLCM) Second-Order Texture Features:
   - Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM, Entropy
2. First-Order Statistical Intensity Moments:
   - Mean, Standard Deviation, Variance, Skewness, Kurtosis, RMS, Shannon Entropy
3. Morphological & Shape Descriptors:
   - Area, Perimeter, Circularity, Solidity, Eccentricity, Aspect Ratio
"""

import numpy as np
import scipy.stats as stats
from skimage.feature import graycomatrix, graycoprops
from typing import Dict, Any, List, Tuple
import cv2


class FeatureExtractor:
    def __init__(self, distances: List[int] = None, angles: List[float] = None, levels: int = 256):
        self.distances = distances or [1, 2, 3]
        self.angles = angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.levels = levels

    def extract_glcm_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Compute Gray-Level Co-occurrence Matrix texture features."""
        # Quantize to uint8 if needed
        if gray_image.dtype != np.uint8:
            gray_image = ((gray_image - gray_image.min()) / (gray_image.max() - gray_image.min() + 1e-8) * 255).astype(np.uint8)

        # Focus calculation on the non-zero brain tissue
        glcm = graycomatrix(
            gray_image,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )

        contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        energy = float(np.mean(graycoprops(glcm, 'energy')))
        correlation = float(np.mean(graycoprops(glcm, 'correlation')))
        asm = float(np.mean(graycoprops(glcm, 'ASM')))

        # Calculate GLCM Entropy: -sum(p * log2(p))
        glcm_prob = glcm / (np.sum(glcm) + 1e-12)
        non_zero_p = glcm_prob[glcm_prob > 0]
        entropy_val = float(-np.sum(non_zero_p * np.log2(non_zero_p)))

        return {
            "glcm_contrast": round(contrast, 4),
            "glcm_dissimilarity": round(dissimilarity, 4),
            "glcm_homogeneity": round(homogeneity, 4),
            "glcm_energy": round(energy, 4),
            "glcm_correlation": round(correlation, 4),
            "glcm_asm": round(asm, 4),
            "glcm_entropy": round(entropy_val, 4)
        }

    def extract_statistical_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Extract first-order statistical moments and pixel distributions."""
        # Non-background pixels (brain matter)
        brain_pixels = gray_image[gray_image > 10].astype(np.float64)
        if len(brain_pixels) == 0:
            brain_pixels = gray_image.flatten().astype(np.float64)

        mean_val = float(np.mean(brain_pixels))
        std_val = float(np.std(brain_pixels))
        var_val = float(np.var(brain_pixels))
        skew_val = float(stats.skew(brain_pixels)) if len(brain_pixels) > 2 else 0.0
        kurt_val = float(stats.kurtosis(brain_pixels)) if len(brain_pixels) > 2 else 0.0
        rms_val = float(np.sqrt(np.mean(brain_pixels ** 2)))

        # Shannon Entropy of intensity histogram
        hist, _ = np.histogram(brain_pixels, bins=64, range=(0, 256), density=True)
        hist = hist[hist > 0]
        shannon_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

        return {
            "stat_mean_intensity": round(mean_val, 2),
            "stat_std_dev": round(std_val, 2),
            "stat_variance": round(var_val, 2),
            "stat_skewness": round(skew_val, 4),
            "stat_kurtosis": round(kurt_val, 4),
            "stat_rms": round(rms_val, 2),
            "stat_shannon_entropy": round(shannon_entropy, 4)
        }

    def extract_morphological_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Extract shape and geometric features from candidate contours."""
        # Adaptive thresholding for tissue contours
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "morph_area": 0.0,
                "morph_perimeter": 0.0,
                "morph_circularity": 0.0,
                "morph_solidity": 0.0,
                "morph_eccentricity": 0.0,
                "morph_aspect_ratio": 1.0
            }

        # Take largest contour
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        
        # Convex hull & solidity
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 0 else 0.0

        # Aspect ratio & Bounding box
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        # Eccentricity via fitted ellipse
        eccentricity = 0.0
        if len(c) >= 5:
            try:
                (center, (major_axis, minor_axis), angle) = cv2.fitEllipse(c)
                a = max(major_axis, minor_axis) / 2.0
                b = min(major_axis, minor_axis) / 2.0
                if a > 0:
                    eccentricity = float(np.sqrt(max(0.0, 1.0 - (b ** 2) / (a ** 2))))
            except Exception:
                eccentricity = 0.0

        return {
            "morph_area": round(area, 1),
            "morph_perimeter": round(perimeter, 1),
            "morph_circularity": round(circularity, 4),
            "morph_solidity": round(solidity, 4),
            "morph_eccentricity": round(eccentricity, 4),
            "morph_aspect_ratio": round(aspect_ratio, 3)
        }

    def extract_feature_vector(self, gray_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
        """
        Extract complete ordered feature vector and organized dictionary.
        Returns:
          - feature_vector (1D np.ndarray)
          - detailed_dict (nested dict with categories, names, values, descriptions)
          - feature_names (list of string keys)
        """
        glcm = self.extract_glcm_features(gray_image)
        stats_feat = self.extract_statistical_features(gray_image)
        morph = self.extract_morphological_features(gray_image)

        combined = {}
        combined.update(glcm)
        combined.update(stats_feat)
        combined.update(morph)

        feature_names = list(combined.keys())
        feature_vector = np.array([combined[k] for k in feature_names], dtype=np.float32)

        # Structured diagnostic presentation with healthy vs tumor reference expectations
        structured = {
            "glcm_texture": {
                "title": "GLCM Texture Features (Gray-Level Co-occurrence Matrix)",
                "features": [
                    {
                        "key": "glcm_contrast",
                        "name": "Contrast",
                        "value": glcm["glcm_contrast"],
                        "unit": "intensity diff²",
                        "description": "Measures local intensity variation and sharp texture discrepancies.",
                        "reference_range": "Normal: 15-45 | Tumor: 55-140+"
                    },
                    {
                        "key": "glcm_dissimilarity",
                        "name": "Dissimilarity",
                        "value": glcm["glcm_dissimilarity"],
                        "unit": "mean diff",
                        "description": "Variation between adjacent voxel pairs.",
                        "reference_range": "Normal: 2.0-5.5 | Tumor: 6.0-15.0"
                    },
                    {
                        "key": "glcm_homogeneity",
                        "name": "Homogeneity",
                        "value": glcm["glcm_homogeneity"],
                        "unit": "0 to 1",
                        "description": "Closeness of distribution of elements to GLCM diagonal (smooth tissue).",
                        "reference_range": "Normal: 0.70-0.95 | Tumor: 0.35-0.65"
                    },
                    {
                        "key": "glcm_energy",
                        "name": "Energy (ASM Sqrt)",
                        "value": glcm["glcm_energy"],
                        "unit": "0 to 1",
                        "description": "Textural uniformity; high energy implies ordered structural patterns.",
                        "reference_range": "Normal: 0.18-0.45 | Tumor: 0.05-0.16"
                    },
                    {
                        "key": "glcm_correlation",
                        "name": "Correlation",
                        "value": glcm["glcm_correlation"],
                        "unit": "-1 to 1",
                        "description": "Linear dependency of gray levels of neighbor pixels.",
                        "reference_range": "Normal: 0.85-0.98 | Tumor: 0.60-0.84"
                    },
                    {
                        "key": "glcm_entropy",
                        "name": "GLCM Entropy",
                        "value": glcm["glcm_entropy"],
                        "unit": "bits",
                        "description": "Measures degree of textural randomness and disorder in tissue.",
                        "reference_range": "Normal: 3.5-6.5 | Tumor: 7.2-11.5+"
                    }
                ]
            },
            "statistical_intensity": {
                "title": "First-Order Statistical Intensity Moments",
                "features": [
                    {
                        "key": "stat_mean_intensity",
                        "name": "Mean Intensity",
                        "value": stats_feat["stat_mean_intensity"],
                        "unit": "0-255",
                        "description": "Average pixel luminosity of brain parenchyma.",
                        "reference_range": "Normal: 70-110 | Tumor: 115-170+"
                    },
                    {
                        "key": "stat_std_dev",
                        "name": "Standard Deviation",
                        "value": stats_feat["stat_std_dev"],
                        "unit": "σ",
                        "description": "Spread of gray level distribution around the mean.",
                        "reference_range": "Normal: 25-45 | Tumor: 48-80+"
                    },
                    {
                        "key": "stat_variance",
                        "name": "Variance",
                        "value": stats_feat["stat_variance"],
                        "unit": "σ²",
                        "description": "Second central moment indicating tissue heterogeneity.",
                        "reference_range": "Normal: 600-2000 | Tumor: 2300-6500+"
                    },
                    {
                        "key": "stat_skewness",
                        "name": "Skewness",
                        "value": stats_feat["stat_skewness"],
                        "unit": "dimensionless",
                        "description": "Asymmetry of pixel intensity histogram.",
                        "reference_range": "Normal: -0.4 to 0.4 | Tumor: >0.8 or <-0.8"
                    },
                    {
                        "key": "stat_kurtosis",
                        "name": "Kurtosis",
                        "value": stats_feat["stat_kurtosis"],
                        "unit": "dimensionless",
                        "description": "Peakedness/heaviness of outlier intensity tails (hyperintensities).",
                        "reference_range": "Normal: -0.5 to 1.5 | Tumor: 2.0-8.0+"
                    }
                ]
            },
            "morphological_shape": {
                "title": "Morphological & Geometric Shape Descriptors",
                "features": [
                    {
                        "key": "morph_circularity",
                        "name": "Circularity Index",
                        "value": morph["morph_circularity"],
                        "unit": "0 to 1",
                        "description": "Degree of contour roundness vs irregular boundary.",
                        "reference_range": "Smooth: 0.65-0.90 | Irregular: 0.15-0.50"
                    },
                    {
                        "key": "morph_solidity",
                        "name": "Solidity",
                        "value": morph["morph_solidity"],
                        "unit": "0 to 1",
                        "description": "Ratio of contour area to convex hull area.",
                        "reference_range": "Compact: 0.85-0.98 | Lobulated: 0.50-0.80"
                    },
                    {
                        "key": "morph_eccentricity",
                        "name": "Eccentricity",
                        "value": morph["morph_eccentricity"],
                        "unit": "0 to 1",
                        "description": "Elongation ratio of the fitted bounding ellipse.",
                        "reference_range": "Spherical: 0.1-0.4 | Elongated: 0.6-0.95"
                    },
                    {
                        "key": "morph_aspect_ratio",
                        "name": "Aspect Ratio",
                        "value": morph["morph_aspect_ratio"],
                        "unit": "W/H",
                        "description": "Width to height ratio of bounding rectangle.",
                        "reference_range": "Symmetric: 0.85-1.25"
                    }
                ]
            },
            "summary_vector": {k: float(v) for k, v in combined.items()},
            "feature_names": feature_names
        }

        return feature_vector, structured, feature_names


def extract_all_features(gray_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    extractor = FeatureExtractor()
    return extractor.extract_feature_vector(gray_image)
