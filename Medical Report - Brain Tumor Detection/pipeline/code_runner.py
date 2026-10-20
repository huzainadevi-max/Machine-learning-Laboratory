"""
NeuroScan AI - In-App Python Code Execution Sandbox
Allows users to type code in the website editor, compile it, run it in real-time,
and capture stdout, stderr, execution time, and any generated Matplotlib plots.
"""

import sys
import io
import time
import traceback
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Non-interactive headless backend for web rendering
import matplotlib.pyplot as plt
from typing import Dict, Any

from .preprocessor import ImagePreprocessor, preprocess_mri
from .features import FeatureExtractor, extract_all_features
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .synthetic_data import DatasetManager


def execute_python_code(code_str: str, timeout_seconds: int = 15) -> Dict[str, Any]:
    """
    Compile and execute arbitrary Python script string, capturing stdout, stderr, and matplotlib plots.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Clear any leftover figures
    plt.close('all')

    t0 = time.time()
    success = True
    error_msg = None
    generated_plots = []

    # Prepare injected execution scope
    def load_sample(tumor_type="glioma"):
        return DatasetManager.generate_realistic_mri_image(tumor_type=tumor_type)

    custom_globals = {
        "__name__": "__main__",
        "np": np,
        "numpy": np,
        "cv2": cv2,
        "plt": plt,
        "matplotlib": matplotlib,
        "ImagePreprocessor": ImagePreprocessor,
        "preprocess_mri": preprocess_mri,
        "FeatureExtractor": FeatureExtractor,
        "extract_all_features": extract_all_features,
        "ModelTrainer": ModelTrainer,
        "ModelEvaluator": ModelEvaluator,
        "DatasetManager": DatasetManager,
        "load_sample_mri": load_sample,
    }

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Compile first to catch syntax errors
        compiled_code = compile(code_str, "<interactive_code_sandbox>", "exec")
        
        # Execute code in isolated namespace
        exec(compiled_code, custom_globals)

        # Capture any Matplotlib plots created during script execution
        fig_nums = plt.get_fignums()
        for fignum in fig_nums:
            fig = plt.figure(fignum)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#131722', edgecolor='none')
            buf.seek(0)
            img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
            generated_plots.append(img_b64)
            plt.close(fig)

    except Exception as e:
        success = False
        error_msg = traceback.format_exc()
        sys.stderr.write(f"\nExecution Error:\n{error_msg}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        plt.close('all')

    execution_duration = round((time.time() - t0) * 1000, 2)
    output_text = stdout_capture.getvalue()
    err_text = stderr_capture.getvalue()

    return {
        "success": success,
        "stdout": output_text,
        "stderr": err_text,
        "error_message": error_msg,
        "execution_time_ms": execution_duration,
        "generated_plots": generated_plots,
        "plots_count": len(generated_plots)
    }
