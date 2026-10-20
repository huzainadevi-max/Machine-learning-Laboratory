"""
NeuroScan AI - Brain Tumor MRI Preprocessor Module
Implements:
1. Grayscale Conversion
2. Gaussian & Bilateral Denoising
3. Skull-Stripping via Otsu Thresholding & Morphological Operations
4. Extreme Contour Brain Cropping
5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
6. Morphological Edge & Contour Extraction
7. Candidate Tumor ROI Localization & Overlay Mask
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple, Optional


class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    @staticmethod
    def to_base64(img_np: np.ndarray) -> str:
        """Convert a NumPy image (BGR, RGB, or Grayscale) to Base64 PNG data URL."""
        if len(img_np.shape) == 2:
            # Grayscale
            _, buffer = cv2.imencode('.png', img_np)
        elif img_np.shape[2] == 3:
            # Check if RGB/BGR
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.dtype == np.uint8 else img_np
            _, buffer = cv2.imencode('.png', bgr)
        else:
            _, buffer = cv2.imencode('.png', img_np)
        return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

    def crop_brain_contour(self, image: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Find extreme contours to crop the non-brain background out."""
        # Blur and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find largest contour (the brain skull outline)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, gray, {"x": 0, "y": 0, "w": image.shape[1], "h": image.shape[0]}

        c = max(contours, key=cv2.contourArea)
        # Find the extreme points
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])

        # Add small margin
        margin = 4
        x1 = max(0, ext_left[0] - margin)
        x2 = min(image.shape[1], ext_right[0] + margin)
        y1 = max(0, ext_top[1] - margin)
        y2 = min(image.shape[0], ext_bot[1] + margin)

        if x2 <= x1 or y2 <= y1:
            return image, gray, {"x": 0, "y": 0, "w": image.shape[1], "h": image.shape[0]}

        cropped_img = image[y1:y2, x1:x2]
        cropped_gray = gray[y1:y2, x1:x2]
        coords = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
        return cropped_img, cropped_gray, coords

    def skull_strip(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Skull stripping using Otsu thresholding + connected component morphological filter."""
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological opening to remove small bridges and skull fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Keep largest connected component (the brain tissue)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[labels == largest_label] = 255
            # Dilate mask slightly to preserve boundary tissue
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = opened

        stripped = cv2.bitwise_and(gray, gray, mask=mask)
        return stripped, mask

    def enhance_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def segment_candidate_tumor(self, enhanced_gray: np.ndarray, original_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Segment potential tumor hyper-intensity region and generate bounding overlay."""
        # Median filter to smooth intensity within brain
        median = cv2.medianBlur(enhanced_gray, 5)
        
        # Adaptive high-intensity thresholding for abnormal hyper-dense tumor tissue
        # Typically tumor tissue in T1-Gd / T2 / FLAIR exhibits bright hyperintensity
        mean_val = np.mean(median[median > 15]) if np.any(median > 15) else 100
        std_val = np.std(median[median > 15]) if np.any(median > 15) else 30
        tumor_thresh_val = min(235, max(140, int(mean_val + 1.2 * std_val)))
        
        _, raw_mask = cv2.threshold(median, tumor_thresh_val, 255, cv2.THRESH_BINARY)
        
        # Morphological opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tumor_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours of candidate tumor
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        overlay = original_rgb.copy()
        tumor_info = {
            "has_candidate": False,
            "area_pixels": 0,
            "bounding_box": None,
            "center": None,
            "perimeter": 0.0,
            "circularity": 0.0,
            "density_score": 0.0
        }
        
        # Filter realistic tumor candidate contours (ignore tiny noise < 100px or full brain > 35% area)
        valid_contours = []
        total_brain_area = np.count_nonzero(enhanced_gray > 15)
        
        for c in contours:
            area = cv2.contourArea(c)
            if 80 < area < (total_brain_area * 0.40 if total_brain_area > 0 else 50000):
                valid_contours.append((c, area))
                
        if valid_contours:
            # Pick largest candidate contour
            best_contour, max_area = max(valid_contours, key=lambda x: x[1])
            x, y, w, h = cv2.boundingRect(best_contour)
            perimeter = cv2.arcLength(best_contour, True)
            circularity = 4 * np.pi * (max_area / (perimeter * perimeter)) if perimeter > 0 else 0
            
            # Draw semi-transparent colored overlay (Red/Amber highlight)
            color_mask = np.zeros_like(original_rgb)
            cv2.drawContours(color_mask, [best_contour], -1, (255, 30, 80), -1)
            overlay = cv2.addWeighted(overlay, 0.75, color_mask, 0.45, 0)
            
            # Draw contour boundary and bounding box
            cv2.drawContours(overlay, [best_contour], -1, (0, 255, 200), 2)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 215, 0), 2)
            cv2.putText(overlay, "ROI (Tumor Candidate)", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
            
            # Calculate tumor density score
            roi_pixels = enhanced_gray[y:y+h, x:x+w]
            density_score = float(np.mean(roi_pixels)) if roi_pixels.size > 0 else 0.0

            tumor_info = {
                "has_candidate": True,
                "area_pixels": int(max_area),
                "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "center": {"x": int(x + w / 2), "y": int(y + h / 2)},
                "perimeter": round(float(perimeter), 2),
                "circularity": round(float(circularity), 3),
                "density_score": round(density_score, 2),
                "roi_brain_ratio_percent": round((max_area / total_brain_area * 100) if total_brain_area > 0 else 0, 2)
            }
            
        return overlay, tumor_mask, tumor_info

    def process(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Execute full end-to-end preprocessing pipeline.
        Input: image_np (H, W, 3) or (H, W) uint8
        Returns dictionary of all stages with base64 representations, metadata, and final preprocessed image.
        """
        # Ensure 3-channel RGB representation
        if len(image_np.shape) == 2:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            gray = image_np.copy()
        elif image_np.shape[2] == 4:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        elif image_np.shape[2] == 3:
            rgb = image_np.copy()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image_np.shape}")

        orig_h, orig_w = gray.shape[:2]

        # Stage 1: Extreme Contour Brain Cropping
        cropped_rgb, cropped_gray, crop_coords = self.crop_brain_contour(rgb, gray)

        # Stage 2: Resize to standardized target size
        resized_rgb = cv2.resize(cropped_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        resized_gray = cv2.resize(cropped_gray, self.target_size, interpolation=cv2.INTER_AREA)

        # Stage 3: Noise reduction (Gaussian & Bilateral Denoising)
        denoised_gray = cv2.bilateralFilter(resized_gray, d=7, sigmaColor=50, sigmaSpace=50)

        # Stage 4: Skull Stripping
        skull_stripped, skull_mask = self.skull_strip(denoised_gray)

        # Stage 5: CLAHE Enhancement
        clahe_enhanced = self.enhance_clahe(skull_stripped)

        # Stage 6: Morphological Edge / Gradient Detection (Sobel + Canny)
        edges = cv2.Canny(clahe_enhanced, 50, 150)
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_grad = cv2.morphologyEx(clahe_enhanced, cv2.MORPH_GRADIENT, kernel_morph)

        # Stage 7: Candidate Tumor Segmentation & Overlay
        overlay_rgb, tumor_mask, tumor_info = self.segment_candidate_tumor(clahe_enhanced, resized_rgb)

        return {
            "metadata": {
                "original_dimensions": f"{orig_w}x{orig_h}",
                "target_dimensions": f"{self.target_size[0]}x{self.target_size[1]}",
                "crop_bounding_box": crop_coords,
                "mean_intensity": round(float(np.mean(resized_gray)), 2),
                "std_intensity": round(float(np.std(resized_gray)), 2),
                "skull_stripped_pixels": int(np.count_nonzero(skull_mask)),
            },
            "tumor_candidate": tumor_info,
            "images": {
                "original": self.to_base64(rgb),
                "cropped": self.to_base64(resized_rgb),
                "grayscale_denoised": self.to_base64(denoised_gray),
                "skull_mask": self.to_base64(skull_mask),
                "skull_stripped": self.to_base64(skull_stripped),
                "clahe_enhanced": self.to_base64(clahe_enhanced),
                "edges_contours": self.to_base64(edges),
                "morph_gradient": self.to_base64(morph_grad),
                "tumor_segmented_mask": self.to_base64(tumor_mask),
                "tumor_overlay": self.to_base64(overlay_rgb),
            },
            "processed_gray": clahe_enhanced,
            "processed_rgb": resized_rgb
        }


def preprocess_mri(image_np: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> Dict[str, Any]:
    preprocessor = ImagePreprocessor(target_size=target_size)
    return preprocessor.process(image_np)
