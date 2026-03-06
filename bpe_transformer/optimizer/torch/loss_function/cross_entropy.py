from einops import rearrange
import torch


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Apply cross-entropy loss on logits without explicitly calling softmax function.
    """
    logits_flat = rearrange(logits, "... vocab_size -> (...) vocab_size")
    target_flat = rearrange(target, "... -> (...)")

    # 1. Subtract max value from logits for numerical stability
    max_val = logits_flat.max(dim=-1, keepdim=True)[0]
    stable_logits = logits_flat - max_val

    # 2. calculate max - log(sum(exp(stable_logits)))
    sum_exp = torch.exp(stable_logits).sum(dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)

    log_probs = stable_logits - log_sum_exp

    # 3. Compare to target: get probability
    log_prob_targets = log_probs[torch.arange(logits_flat.shape[0], device=logits_flat.device), target_flat]

    loss = -log_prob_targets.mean()

    return loss
