"""
End-to-end inference demo for Samsung PRISM Layer-1 pipeline.

Usage:
    python src/inference_demo.py <image_path>

Example:
    python src/inference_demo.py test_images/fridge.jpg
"""

import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
from src.preprocess import load_frame
from src.router import RouterMLP, select_expert, EXPERT_NAMES, CLASS_NAMES
from src.nms_utils import apply_nms          # Person D writes this

# ── Config ────────────────────────────────────────────────────────────
CONF_THRESHOLD    = 0.60
ENTROPY_THRESHOLD = 0.80
EXPERT_CONF_ACCEPT  = 0.70
EXPERT_CONF_UNCERTAIN = 0.40

# ── Load backbone ─────────────────────────────────────────────────────
print("Loading backbone...")
backbone = torchvision.models.mobilenet_v3_small(weights=None)
backbone.load_state_dict(
    torch.load("models/backbone.pt", map_location="cpu")
)
backbone.eval()

# ── Load router ───────────────────────────────────────────────────────
print("Loading router...")
router = RouterMLP(num_experts=3)
router.load_state_dict(
    torch.load("models/router.pt", map_location="cpu")
)
router.eval()

# ── Load expert heads ─────────────────────────────────────────────────
# Person C sends expert_0 and expert_1
# Person D sends expert_2
print("Loading expert heads...")
expert_heads = {
    0: torch.load("models/expert_0_display.pt",  map_location="cpu"),
    1: torch.load("models/expert_1_kitchen.pt",   map_location="cpu"),
    2: torch.load("models/expert_2_climate.pt",   map_location="cpu"),
}
for e in expert_heads.values():
    e.eval()

print("All models loaded.\n")

# ── GAP extraction helper ─────────────────────────────────────────────
def extract_gap(frame: torch.Tensor) -> torch.Tensor:
    """
    Run frame through backbone.features, then Global Average Pool.
    Returns: gap vector of shape [1, 960]
    """
    with torch.no_grad():
        feature_maps = backbone.features(frame)          # [1, 960, 10, 10]
        gap = torch.nn.functional.adaptive_avg_pool2d(feature_maps, 1)
        gap = gap.view(1, -1)                            # [1, 960]
    return gap

# ── Main inference function ───────────────────────────────────────────
def run_inference(image_path: str) -> dict:
    print(f"{'='*55}")
    print(f"Image: {image_path}")
    t0 = time.time()

    # Stage 1: Preprocess
    frame = load_frame(image_path)               # [1, 3, 320, 320]
    print(f"Stage 1 ✅ Frame preprocessed: {frame.shape}")

    # Stage 2: Backbone → GAP vector
    gap = extract_gap(frame)                     # [1, 960]
    print(f"Stage 2 ✅ GAP vector extracted: {gap.shape}")

    # Stage 3: Router logits
    with torch.no_grad():
        logits = router(gap)                     # [1, 3]
    probs = torch.softmax(logits, dim=-1).squeeze()
    print(f"Stage 3 ✅ Router probs: "
          f"Display={probs[0]:.2f} | "
          f"Kitchen={probs[1]:.2f} | "
          f"Climate={probs[2]:.2f}")

    # Stage 4: Expert selection
    expert_id, status, confidence = select_expert(
        logits, CONF_THRESHOLD, ENTROPY_THRESHOLD
    )

    if status == "UNCERTAIN":
        print(f"Stage 4 ⚠️  UNCERTAIN (max_prob={confidence:.2f}) — rescan needed")
        return {"status": "UNCERTAIN", "label": None, "confidence": confidence}

    print(f"Stage 4 ✅ Expert selected: {expert_id} ({EXPERT_NAMES[expert_id]}) "
          f"| confidence={confidence:.2f}")

    # Stage 5: Expert inference
    expert = expert_heads[expert_id]
    with torch.no_grad():
        raw_output = expert(frame)

    # ── How to handle raw_output depends on Person C/D's export format.
    # If they export a full torchvision SSD model, output is a list of dicts.
    # Print raw output first time to confirm format:
    print(f"Stage 5 ✅ Expert raw output type: {type(raw_output)}")

    # Torchvision SSD returns: [{'boxes': Tensor, 'labels': Tensor, 'scores': Tensor}]
    if isinstance(raw_output, (list, tuple)) and isinstance(raw_output[0], dict):
        boxes  = raw_output[0]['boxes']     # [N, 4] in xyxy format
        scores = raw_output[0]['scores']    # [N]
        labels = raw_output[0]['labels']    # [N]
    else:
        # Fallback: print and stop — coordinate with Person C/D
        print("⚠️  Unexpected expert output format. Raw output:")
        print(raw_output)
        return {"status": "FORMAT_ERROR", "raw": str(raw_output)}

    # Stage 6: NMS + confidence filtering
    boxes, scores, labels = apply_nms(boxes, scores, labels,
                                      iou_threshold=0.45,
                                      score_threshold=EXPERT_CONF_ACCEPT)

    if len(scores) == 0:
        print("Stage 6 ⚠️  No detections above threshold.")
        return {"status": "NO_DETECTION", "label": None}

    # Stage 7: Build output
    detections = []
    for box, score, label_id in zip(boxes, scores, labels):
        det = {
            "label":       CLASS_NAMES.get(label_id.item(), f"class_{label_id.item()}"),
            "confidence":  round(score.item(), 4),
            "bbox":        [round(v, 4) for v in box.tolist()],  # [x1, y1, x2, y2]
            "expert_id":   expert_id,
            "expert_name": EXPERT_NAMES[expert_id],
        }
        detections.append(det)

    t1 = time.time()
    latency_ms = round((t1 - t0) * 1000, 1)

    result = {
        "status":     "ACCEPT",
        "detections": detections,
        "latency_ms": latency_ms,
    }

    print(f"Stage 7 ✅ Output:")
    print(json.dumps(result, indent=2))
    print(f"Total latency: {latency_ms} ms")
    return result


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference_demo.py <image_path>")
        sys.exit(1)
    run_inference(sys.argv[1])