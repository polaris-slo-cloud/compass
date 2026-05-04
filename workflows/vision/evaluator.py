"""Vision evaluator: AP@0.5 over COCO val2017 for COMPASS-V."""

import json
import os
from typing import Any, Dict, List

import numpy as np

from compass.search.evaluator import Evaluator
from workflows.vision.pipeline import CascadedDetector, Detection


def _iou_single(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


# COCO category_id (non-contiguous 1..90) -> YOLO class index (0..79).
_COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
_COCO_TO_YOLO = {cat_id: idx for idx, cat_id in enumerate(_COCO_CAT_IDS)}


def _coco_to_yolo_class(coco_cat_id: int) -> int:
    return _COCO_TO_YOLO.get(coco_cat_id, -1)


def _ap50(pred_dets: List[Detection], gt_boxes: List[Dict], iou_threshold: float = 0.5) -> float:
    """11-point interpolated AP@0.5 for a single image."""
    if not gt_boxes:
        return 1.0 if not pred_dets else 0.0
    if not pred_dets:
        return 0.0

    preds_sorted = sorted(pred_dets, key=lambda d: d.confidence, reverse=True)
    gt_matched = [False] * len(gt_boxes)
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, pred in enumerate(preds_sorted):
        best_iou = 0.0
        best_gt = -1
        for j, gt in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            gx1, gy1 = gt["bbox"][0], gt["bbox"][1]
            gx2 = gx1 + gt["bbox"][2]
            gy2 = gy1 + gt["bbox"][3]
            if pred.class_id != _coco_to_yolo_class(gt["category_id"]):
                continue
            iou = _iou_single(pred.bbox, [gx1, gy1, gx2, gy2])
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold and best_gt >= 0:
            tp[i] = 1
            gt_matched[best_gt] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / len(gt_boxes)
    precision = cum_tp / (cum_tp + cum_fp)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_recall = precision[recall >= t]
        ap += (prec_at_recall.max() if len(prec_at_recall) > 0 else 0) / 11
    return float(ap)


class VisionEvaluator(Evaluator):
    """Evaluates cascaded detector configs on COCO val2017 (mAP@0.5)."""

    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        image_ids: List[int] = None,
    ):
        self.images_dir = images_dir
        self.detector = CascadedDetector()

        with open(annotations_path) as f:
            coco_data = json.load(f)

        self.anns_by_image: Dict[int, List[Dict]] = {}
        for ann in coco_data["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)
        self.image_info = {img["id"]: img for img in coco_data["images"]}

        self.image_ids = image_ids if image_ids is not None else sorted(self.image_info)

    @property
    def n_samples(self) -> int:
        return len(self.image_ids)

    def evaluate_partial(
        self, config: Dict[str, Any], indices: List[int]
    ) -> List[float]:
        scores = []
        for idx in indices:
            img_id = self.image_ids[idx % len(self.image_ids)]
            img_info = self.image_info[img_id]
            img_path = os.path.join(self.images_dir, img_info["file_name"])
            gt_anns = [a for a in self.anns_by_image.get(img_id, []) if not a.get("iscrowd", 0)]
            try:
                preds = self.detector.detect(img_path, config)
                ap = _ap50(preds, gt_anns)
            except Exception:
                ap = 0.0
            scores.append(ap)
        return scores
