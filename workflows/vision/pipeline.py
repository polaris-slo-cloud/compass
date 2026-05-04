"""
Cascaded object detection pipeline with confidence routing.

Architecture:
  Input Image -> [Stage 1: Fast Detector] -> conf >= threshold? -> Accept
                                           -> conf < threshold  -> [Stage 2: Heavy Verifier]
                                                                      -> Merge detections
                 -> [Stage 3: NMS Post-processing] -> Final Detections
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int


# Global model cache
_model_cache: Dict[str, YOLO] = {}


def get_model(model_name: str) -> YOLO:
    """Load and cache YOLO model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(f"{model_name}.pt")
    return _model_cache[model_name]


def nms_merge(
    detections: List[Detection], iou_threshold: float
) -> List[Detection]:
    """Apply NMS to merge detections from multiple stages."""
    if not detections:
        return []

    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_id for d in detections])

    keep = []
    # NMS per class
    for cls_id in np.unique(classes):
        cls_mask = classes == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        # Sort by confidence
        order = cls_scores.argsort()[::-1]
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        cls_indices = cls_indices[order]

        while len(cls_boxes) > 0:
            keep.append(cls_indices[0])
            if len(cls_boxes) == 1:
                break

            # IoU with rest
            ious = _compute_iou(cls_boxes[0], cls_boxes[1:])
            mask = ious < iou_threshold
            cls_boxes = cls_boxes[1:][mask]
            cls_scores = cls_scores[1:][mask]
            cls_indices = cls_indices[1:][mask]

    return [detections[i] for i in keep]


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection

    return intersection / np.maximum(union, 1e-6)


class CascadedDetector:
    """Confidence-routed cascaded detection pipeline."""

    def detect(self, image_path: str, config: Dict[str, Any]) -> List[Detection]:
        """Run cascaded detection on a single image."""
        detector_model = config["detector_model"]
        conf_threshold = config["conf_threshold"]
        verifier_model = config["verifier_model"]
        nms_threshold = config["nms_threshold"]

        # Stage 1: Fast detector (low conf to get all candidates)
        detector = get_model(detector_model)
        results = detector(image_path, verbose=False, conf=0.01)

        stage1_accept = []
        stage1_uncertain = []

        for r in results:
            if r.boxes is None:
                continue
            for box, conf, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                det = Detection(
                    bbox=box.tolist(),
                    confidence=float(conf),
                    class_id=int(cls),
                )
                if conf >= conf_threshold:
                    stage1_accept.append(det)
                else:
                    stage1_uncertain.append(det)

        # Stage 2: Heavy verifier for uncertain detections
        stage2_dets = []
        if verifier_model != "none" and stage1_uncertain:
            verifier = get_model(verifier_model)
            v_results = verifier(image_path, verbose=False, conf=conf_threshold)
            for r in v_results:
                if r.boxes is None:
                    continue
                for box, conf, cls in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                ):
                    stage2_dets.append(Detection(
                        bbox=box.tolist(),
                        confidence=float(conf),
                        class_id=int(cls),
                    ))

        # Stage 3: NMS merge
        all_dets = stage1_accept + stage2_dets
        final = nms_merge(all_dets, nms_threshold)
        return final
