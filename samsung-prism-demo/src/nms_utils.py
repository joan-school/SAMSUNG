# src/nms_utils.py  — Person D writes this
import torch
from torchvision.ops import nms

def apply_nms(boxes, scores, labels, iou_threshold=0.45, score_threshold=0.55):
    """
    Args:
        boxes:           Tensor [N, 4]  xyxy format
        scores:          Tensor [N]
        labels:          Tensor [N]
        iou_threshold:   float
        score_threshold: float
    Returns:
        filtered boxes, scores, labels as Tensors
    """
    # Filter by score threshold first
    keep_mask = scores >= score_threshold
    boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]

    if len(scores) == 0:
        return boxes, scores, labels

    # NMS
    keep_idx = nms(boxes, scores, iou_threshold)
    return boxes[keep_idx], scores[keep_idx], labels[keep_idx]