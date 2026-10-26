"""
LUFFY core algorithm functions for on/off-policy learning.
Adapted from LUFFY (https://github.com/EleutherAI/luffy) to work with the
current verl API. Key changes:
  - eos_mask -> response_mask parameter naming
  - Uses verl.utils.torch_functional.masked_mean
"""

from collections import defaultdict

import torch

import verl.utils.torch_functional as verl_F


def compute_sft_pure_loss(log_prob, response_mask):
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(sft_losses, response_mask)
    return sft_loss


def compute_grpo_outcome_advantage_split(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    on_policy_mask: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    """
    GRPO advantage that normalizes only over on-policy samples per prompt group.
    Off-policy (teacher-prefix) samples still receive the broadcast advantage
    but do not participate in the mean/std calculation.
    """
    response_length = token_level_rewards.shape[-1]
    scores = (token_level_rewards * response_mask).sum(dim=-1)

    id2score: dict[str, list] = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return scores, scores


def compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange,
    clip_upper_bound,
    prefix_mask,
    off_cliprange,
    off_normalize=False,
    off_abs_cliprange=None,
    off_max_clip=None,
    off_min_clip=None,
    all_max_clip=None,
    off_policy_reshape="no_reshape",
    off_policy_reshape_weight=1.0,
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape",
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_agg_mode="token-mean",
):
    """Token-level on/off-policy loss for LUFFY.

    On-policy tokens use standard PPO clipped ratio.
    Off-policy (teacher prefix) tokens use reshaped / importance-weighted ratio.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # --- on-policy ratio ---
    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl)
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.sqrt(torch.exp(negative_approx_kl))
    elif on_policy_reshape == "pow":
        ratio = torch.pow(torch.exp(negative_approx_kl), on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        ratio = (prob / (prob + 0.1)) / (old_prob / (old_prob + 0.1))
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        ratio = (prob / (prob + 0.5)) / (old_prob / (old_prob + 0.5))
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)

    on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), response_mask)
    on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * response_mask)

    # --- off-policy ratio ---
    if target_probs is None:
        off_ratio = torch.exp(log_prob)
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape in ("p_div_p_0.1", "p_div_p_0.5", "p_div_p_0.3"):
            c = float(off_policy_reshape.split("_")[-1])
            off_ratio = off_ratio / (off_ratio + c)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs + 1e-6)
        off_ratio = off_ratio * prefix_mask

    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean(
            (off_ratio == off_max_clip).float(), prefix_mask * response_mask
        )
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)

    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean(
            (off_ratio == off_min_clip).float(), prefix_mask * response_mask
        )
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * response_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * response_mask)
    if off_pg_loss.isnan().item():
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)

    prefix_mask_f = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask_f + on_pg_losses * (1 - prefix_mask_f)

    off_policy_prob = verl_F.masked_mean(torch.exp(log_prob), prefix_mask_f * response_mask)
    if off_policy_prob.isnan().item():
        off_policy_prob = torch.tensor(0.0)
    on_policy_prob = verl_F.masked_mean(torch.exp(old_log_prob), (1.0 - prefix_mask_f) * response_mask)
    if on_policy_prob.isnan().item():
        on_policy_prob = torch.tensor(0.0)

    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        response_mask = response_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask

    pg_loss = verl_F.masked_mean(pg_losses, response_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }
