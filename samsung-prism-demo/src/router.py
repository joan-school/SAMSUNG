import torch
import torch.nn as nn


class RouterMLP(nn.Module):
    """
    Lightweight MLP that takes a 960-dim GAP vector from MobileNetV3-Small
    and outputs a probability distribution over N experts.

    Architecture: 960 -> 256 -> 128 -> num_experts
    """
    def __init__(self, num_experts: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(960, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, num_experts)   # raw logits, no softmax here
        )

    def forward(self, gap_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gap_vector: shape [B, 960]
        Returns:
            logits: shape [B, num_experts]
        """
        return self.network(gap_vector)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of the softmax distribution.
    Higher entropy = router is more uncertain.
    Args:
        logits: shape [B, num_experts]
    Returns:
        entropy: shape [B]
    """
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    return entropy


def select_expert(
    logits: torch.Tensor,
    conf_threshold: float = 0.60,
    entropy_threshold: float = 0.80
) -> tuple[int, str, float]:
    """
    Given router logits for a single frame, select the expert.
    Args:
        logits: shape [1, num_experts]
    Returns:
        (expert_id, status, confidence)
        status is "CONFIDENT" or "UNCERTAIN"
    """
    probs     = torch.softmax(logits, dim=-1)          # [1, num_experts]
    entropy   = compute_entropy(logits).item()
    max_prob, expert_id = probs.max(dim=-1)
    max_prob  = max_prob.item()
    expert_id = expert_id.item()

    if max_prob >= conf_threshold and entropy <= entropy_threshold:
        return expert_id, "CONFIDENT", max_prob
    else:
        return -1, "UNCERTAIN", max_prob


EXPERT_NAMES = {
    0: "Display",    # TV
    1: "Kitchen",    # Fridge, Microwave
    2: "Climate",    # AC
}

CLASS_NAMES = {
    0: "tv",
    1: "refrigerator",
    2: "microwave",
    3: "air_conditioner",
}