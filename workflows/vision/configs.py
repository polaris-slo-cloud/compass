"""Vision configuration space for COMPASS-V.

The cascaded YOLO detector has 385 valid configurations:
    3 detectors * 7 conf thresholds * 4 verifiers (incl. "none") * 5 NMS thresholds
    constrained so the verifier is strictly heavier than the detector
    (or "none").
"""

from typing import Any, Dict

from compass.search.parameter_space import NormType, ParameterSpace


# Lighter -> heavier; used both as the parameter ordering and for the
# "verifier > detector" constraint.
MODEL_WEIGHT: Dict[str, int] = {
    "yolov8n": 0,
    "yolov8s": 1,
    "yolov8m": 2,
    "yolov8l": 3,
    "yolov8x": 4,
}

DETECTOR_MODELS = ["yolov8n", "yolov8s", "yolov8m"]
CONF_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
VERIFIER_MODELS = ["yolov8m", "yolov8l", "yolov8x", "none"]
NMS_THRESHOLDS = [0.3, 0.45, 0.5, 0.6, 0.7]


def _verifier_heavier_than_detector(config: Dict[str, Any]) -> bool:
    if config["verifier_model"] == "none":
        return True
    return MODEL_WEIGHT[config["verifier_model"]] > MODEL_WEIGHT[config["detector_model"]]


def parameter_space() -> ParameterSpace:
    """Construct the vision parameter space (385 valid configurations)."""
    return ParameterSpace(
        params={
            "detector_model": DETECTOR_MODELS,
            "conf_threshold": CONF_THRESHOLDS,
            "verifier_model": VERIFIER_MODELS,
            "nms_threshold": NMS_THRESHOLDS,
        },
        norm_types={
            "detector_model": NormType.CATEGORICAL,
            "verifier_model": NormType.CATEGORICAL,  # "none" breaks numeric extraction
            "conf_threshold": NormType.LINEAR,
            "nms_threshold": NormType.LINEAR,
        },
        constraints=[_verifier_heavier_than_detector],
    )
