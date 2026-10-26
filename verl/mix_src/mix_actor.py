"""
LUFFY MIX Actor: DataParallelPPOActor with on/off-policy loss support.
Adapted from LUFFY to the current verl DataParallelPPOActor API.

Key API changes vs the original LUFFY actor:
  - ppo_micro_batch_size -> ppo_micro_batch_size_per_gpu
  - _forward_micro_batch now takes (model_inputs_dict, temperature, calculate_entropy)
  - response_mask comes from data["response_mask"] (pre-computed)
  - prepare_dynamic_batch replaces rearrange_micro_batches
  - micro_batch is DataProto; model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
  - compute_policy_loss returns 4 values (added pg_clipfrac_lower)
  - agg_loss helper for entropy/kl aggregation
  - FSDPModule (FSDP2) support in _optimizer_step
"""

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.workers.actor.dp_actor import DataParallelPPOActor

__all__ = ["MIXDataParallelPPOActor"]


class MIXDataParallelPPOActor(DataParallelPPOActor):
    """Actor that supports LUFFY on/off-policy and SFT-multitask loss modes."""

    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config, actor_module, actor_optimizer)

        self.use_adaptive_temperature = self.config.get("use_adaptive_temperature", False)
        self.adaptive_temperature_target_entropy = self.config.get("adaptive_temperature_target_entropy", 0.0)

        if self.use_adaptive_temperature:
            self.log_alpha = torch.tensor(np.log(self.config.entropy_coeff), dtype=torch.float)
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha], lr=self.config.get("alpha_lr", 1e-4), betas=(0.9, 0.999), weight_decay=1e-2
            )
        else:
            self.alpha_optimizer = None

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "prefix_mask",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.get("use_off_policy_loss", False) and self.config.get("use_off_policy_probs", False):
            select_keys.append("target_probs")

        has_multi_modal = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_keys = ["multi_modal_inputs"] if has_multi_modal else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                if self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    prefix_mask = model_inputs["prefix_mask"]

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=(entropy_coeff != 0)
                    )

                    use_sft_multitask = self.config.get("use_sft_multitask_loss", False)
                    use_off_policy = self.config.get("use_off_policy_loss", False)

                    if use_sft_multitask:
                        assert not use_off_policy, "Cannot use both off-policy loss and SFT multitask loss."
                        pg_loss, pg_clipfrac, ppo_kl = self._sft_multitask_loss(
                            log_prob, old_log_prob, advantages, response_mask, prefix_mask, clip_ratio, loss_agg_mode
                        )

                    elif use_off_policy:
                        pg_loss, pg_clipfrac, ppo_kl = self._off_policy_loss(
                            log_prob, old_log_prob, advantages, response_mask, prefix_mask,
                            clip_ratio, loss_agg_mode, model_inputs, micro_batch_metrics
                        )

                    else:
                        pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=self.config.get("clip_ratio_low", None) or clip_ratio,
                            cliprange_high=self.config.get("clip_ratio_high", None) or clip_ratio,
                            clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                            loss_agg_mode=loss_agg_mode,
                        )

                    # entropy
                    if entropy_coeff != 0 and entropy is not None:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    else:
                        entropy_loss = torch.tensor(0.0)

                    if self.use_adaptive_temperature:
                        policy_loss, micro_batch_metrics = self._adaptive_entropy_loss(
                            pg_loss, entropy_loss, entropy, response_mask, micro_batch_metrics
                        )
                    else:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.get("use_ppo_kl_loss", False):
                        policy_loss = policy_loss + ppo_kl.abs() * self.config.get("ppo_kl_loss_coef", 0.0)
                        micro_batch_metrics["actor/ppo_kl_loss"] = ppo_kl.abs().detach().item()

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update({
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    })
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        return metrics

    # ------------------------------------------------------------------ #
    #  Loss helpers
    # ------------------------------------------------------------------ #
    def _sft_multitask_loss(self, log_prob, old_log_prob, advantages, response_mask, prefix_mask, clip_ratio, loss_agg_mode):
        from .mix_core_alg import compute_sft_pure_loss

        off_policy_mask = prefix_mask.any(-1)
        sft_loss = compute_sft_pure_loss(log_prob[off_policy_mask], response_mask[off_policy_mask])
        if log_prob[off_policy_mask].numel() == 0:
            sft_loss = torch.tensor(0.0)

        on_mask = ~off_policy_mask
        on_lp, on_old_lp = log_prob[on_mask], old_log_prob[on_mask]
        on_adv, on_rm = advantages[on_mask], response_mask[on_mask]

        pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
            old_log_prob=on_old_lp, log_prob=on_lp, advantages=on_adv, response_mask=on_rm,
            cliprange=clip_ratio,
            cliprange_low=self.config.get("clip_ratio_low", None) or clip_ratio,
            cliprange_high=self.config.get("clip_ratio_high", None) or clip_ratio,
            clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
            loss_agg_mode=loss_agg_mode,
        )
        sft_coef = self.config.get("sft_loss_coef", 1.0)
        return sft_loss * sft_coef + pg_loss, pg_clipfrac, ppo_kl

    def _off_policy_loss(self, log_prob, old_log_prob, advantages, response_mask, prefix_mask,
                         clip_ratio, loss_agg_mode, model_inputs, micro_batch_metrics):
        from .mix_core_alg import compute_token_on_off_policy_loss

        ret = compute_token_on_off_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=clip_ratio,
            clip_upper_bound=self.config.get("clip_upper_bound", 5.0),
            prefix_mask=prefix_mask,
            off_cliprange=self.config.get("off_policy_cliprange", 0.2),
            off_normalize=self.config.get("off_policy_normalize", False),
            off_max_clip=self.config.get("off_policy_max_clip", -1) if self.config.get("off_policy_max_clip", -1) != -1 else None,
            off_min_clip=self.config.get("off_policy_min_clip", -1) if self.config.get("off_policy_min_clip", -1) != -1 else None,
            all_max_clip=self.config.get("all_max_clip", -1) if self.config.get("all_max_clip", -1) != -1 else None,
            off_policy_reshape=self.config.get("off_policy_reshape", "no_reshape"),
            off_policy_reshape_weight=self.config.get("off_policy_reshape_weight", 1.0),
            off_policy_reshape_pow_exp=self.config.get("off_policy_reshape_pow_exp", 0.5),
            on_policy_reshape=self.config.get("on_policy_reshape", "no_reshape"),
            on_policy_reshape_weight=self.config.get("on_policy_reshape_weight", 1.0),
            on_policy_reshape_pow_exp=self.config.get("on_policy_reshape_pow_exp", 0.5),
            target_probs=model_inputs.get("target_probs", None),
            loss_agg_mode=loss_agg_mode,
        )

        extra = {}
        for k in ("off_pg_loss", "on_pg_loss", "off_pg_clipfrac", "off_policy_prob",
                   "on_policy_prob", "off_ratio_mean", "off_ratio_max_clip_frac", "off_ratio_min_clip_frac"):
            if k in ret:
                extra[f"actor/{k}"] = ret[k].detach().item()
        append_to_dict(micro_batch_metrics, extra)

        return ret["pg_loss"], ret["on_pg_clipfrac"], ret["ppo_kl"]

    def _adaptive_entropy_loss(self, pg_loss, entropy_loss, entropy, response_mask, micro_batch_metrics):
        if self.config.get("use_adaptive_temperature_fixed", False):
            target_entropy = self.adaptive_temperature_target_entropy
            coeff = (target_entropy / (entropy_loss + 1e-8)).detach().item() * self.config.entropy_coeff
            policy_loss = pg_loss - entropy_loss * coeff
            micro_batch_metrics["actor/entropy_coeff"] = coeff
        else:
            target_entropy = self.adaptive_temperature_target_entropy
            coeff = self.log_alpha.exp()
            clip_val = self.config.get("adaptive_temperature_clip", 0)
            if clip_val > 0:
                coeff = torch.clamp(coeff, max=clip_val)
            alpha_loss = verl_F.masked_mean(entropy - target_entropy, response_mask).detach() * coeff
            alpha_loss = alpha_loss / self.gradient_accumulation
            alpha_loss.backward()
            policy_loss = pg_loss - entropy_loss * coeff.detach().item()
            micro_batch_metrics["actor/alpha_loss"] = alpha_loss.detach().item()
            micro_batch_metrics["actor/entropy_coeff"] = coeff.detach().item()
            micro_batch_metrics["actor/log_alpha"] = self.log_alpha.detach().item()
        return policy_loss, micro_batch_metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.step()
        return grad_norm
