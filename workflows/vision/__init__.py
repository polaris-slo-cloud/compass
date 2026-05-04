"""Cascaded YOLO object-detection workflow on COCO val2017."""

from workflows.vision.configs import (
    CONF_THRESHOLDS,
    DETECTOR_MODELS,
    NMS_THRESHOLDS,
    VERIFIER_MODELS,
    parameter_space,
)
from workflows.vision.evaluator import VisionEvaluator
from workflows.vision.pipeline import CascadedDetector, Detection

__all__ = [
    "VisionEvaluator",
    "parameter_space",
    "CascadedDetector",
    "Detection",
    "DETECTOR_MODELS",
    "VERIFIER_MODELS",
    "CONF_THRESHOLDS",
    "NMS_THRESHOLDS",
]
