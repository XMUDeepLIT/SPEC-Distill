"""
LUFFY MIX Trainer: RayPPOTrainer with off-policy knowledge integration.
Adapted from LUFFY to the current verl RayPPOTrainer API.

Key differences from the base trainer:
  1. Uses RLHFDatasetWithTarget (adds tgt_input_ids for teacher traces)
  2. Pops tgt_input_ids into the generation batch
  3. Adds grpo_split advantage estimator (on-policy-only normalisation)
  4. Applies alpha/beta prefix reward weighting on advantages
  5. Computes on/off-policy metrics
"""

import os
import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import compute_timing_metrics, compute_throughout_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean


class MIXRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer extended with LUFFY off-policy knowledge integration."""

    def _create_dataloader(self, train_dataset=None, val_dataset=None, collate_fn=None, train_sampler=None):
        """Override to use RLHFDatasetWithTarget for training."""
        from verl.mix_src.rl_dataset_with_target import RLHFDatasetWithTarget, ResumableRandomSampler
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        if train_dataset is None:
            train_dataset = RLHFDatasetWithTarget(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                config=self.config.data,
                processor=getattr(self, "processor", None),
            )

        if val_dataset is None:
            from verl.trainer.main_ppo import create_rl_dataset
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, getattr(self, "processor", None)
            )

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            if self.config.data.get("shuffle", True):
                train_sampler = ResumableRandomSampler(data_source=self.train_dataset)
            else:
                from torch.utils.data import SequentialSampler
                train_sampler = SequentialSampler(data_source=self.train_dataset)

        if collate_fn is None:
            collate_fn = default_collate_fn

        num_workers = self.config.data.get("dataloader_num_workers", 0)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.get("val_batch_size", None)
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}, val: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps: {e}")

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self.global_steps += 1
        n_samples = self.config.actor_rollout_ref.rollout.n
        reward_impl_version = self.config.data.get("reward_impl_version", 0)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                metrics = {}
                timing_raw = {}

                # pop keys for generation (including tgt_input_ids)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids", "tgt_input_ids"]
                non_tensor_keys_to_pop = []
                if "raw_prompt_ids" in batch.non_tensor_batch:
                    non_tensor_keys_to_pop.append("raw_prompt_ids")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_keys_to_pop,
                )
                # breakpoint()
                gen_batch.meta_info["global_steps"] = self.global_steps

                with marked_timer("step", timing_raw):
                    # generate
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    batch = batch.repeat(repeat_times=n_samples, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "prefix_ratios" in gen_batch_output.meta_info:
                        metrics["batch/avg_prefix_ratio"] = float(
                            np.mean(gen_batch_output.meta_info["prefix_ratios"])
                        )

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # balance batch
                    if self.config.trainer.get("balance_batch", False):
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # reward
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores"] = reward_tensor

                    # on/off policy reward metrics
                    self._log_on_off_metrics(batch, reward_tensor, metrics)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch.get("entropys", None)
                        if entropys is not None:
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            metrics["actor/entropy"] = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            ).detach().item()
                            old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            ref_in_actor = getattr(self, "ref_in_actor", False)
                            if not ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # KL penalty
                        if self.config.algorithm.get("use_kl_in_reward", False):
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # advantage
                        adv_estimator = self.config.algorithm.adv_estimator
                        if adv_estimator == "grpo_split":
                            batch = self._compute_grpo_split_advantage(batch)
                        else:
                            norm_adv_by_std = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=n_samples,
                                norm_adv_by_std_in_grpo=norm_adv_by_std,
                                config=self.config.algorithm,
                            )

                        # alpha/beta prefix reward weighting
                        self._apply_prefix_weighting(batch)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.get(
                                "multi_turn", OmegaConf.create({"enable": False})
                            ).enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    # validate
                    is_last_step = self.global_steps >= self.total_training_steps
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # metrics
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics_luffy(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1

                if is_last_step:
                    return

    # ------------------------------------------------------------------ #
    #  LUFFY-specific helpers
    # ------------------------------------------------------------------ #
    def _compute_grpo_split_advantage(self, batch):
        """GRPO advantage normalized only over on-policy samples."""
        from verl.mix_src.mix_core_alg import compute_grpo_outcome_advantage_split

        token_level_rewards = batch.batch["token_level_rewards"]
        response_mask = batch.batch["response_mask"]
        index = batch.non_tensor_batch["uid"]
        prefix_mask = batch.batch["prefix_mask"]
        on_policy_mask = ~prefix_mask.any(-1)
        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            on_policy_mask=on_policy_mask,
            norm_adv_by_std_in_grpo=norm_adv,
        )
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = returns
        return batch

    def _apply_prefix_weighting(self, batch):
        """Apply alpha/beta weighting on advantages for prefix vs continuation tokens."""
        if "prefix_mask" not in batch.batch:
            return

        prefix_mask = batch.batch["prefix_mask"]
        advantages = batch.batch["advantages"]

        alpha = self.config.actor_rollout_ref.rollout.get("prefix_reward_weight_alpha", 1.0)
        beta = self.config.actor_rollout_ref.rollout.get("prefix_reward_weight_beta", 1.0)
        alpha_weight = prefix_mask.float() * alpha
        beta_weight = (~prefix_mask).float() * beta
        batch.batch["advantages"] = (alpha_weight + beta_weight) * advantages

        if self.config.actor_rollout_ref.actor.get("use_sft_prefix_reward", False):
            n_samples = self.config.actor_rollout_ref.rollout.n
            reward_weight = self.config.actor_rollout_ref.actor.get("sft_prefix_reward_weight", 1.0)
            batch.batch["advantages"][prefix_mask] = reward_weight / n_samples

    def _log_on_off_metrics(self, batch, reward_tensor, metrics):
        """Log on/off-policy solve rates."""
        if "prefix_mask" not in batch.batch:
            return

        prefix_mask = batch.batch["prefix_mask"]
        off_policy_mask = prefix_mask.any(-1)
        on_policy_mask = ~off_policy_mask

        success_value = 1.0
        seq_scores = reward_tensor.sum(-1)
        metrics["batch/solved"] = (seq_scores == success_value).sum().item() / max(len(seq_scores), 1)
        if on_policy_mask.sum() > 0:
            metrics["batch/on_solved"] = (
                (seq_scores[on_policy_mask] == success_value).sum().item() / on_policy_mask.sum().item()
            )
        if off_policy_mask.sum() > 0:
            metrics["batch/off_solved"] = (
                (seq_scores[off_policy_mask] == success_value).sum().item() / off_policy_mask.sum().item()
            )


def compute_data_metrics_luffy(batch, use_critic=True):
    """Compute data metrics with on/off-policy breakdown."""
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]
    response_mask = batch.batch["response_mask"].bool()

    from verl.trainer.ppo.metric_utils import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    metrics = {
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(
            torch.eq(response_length, max_response_length).float()
        ).detach().item(),
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
    }

    if "prefix_mask" in batch.batch:
        off_policy_mask = batch.batch["prefix_mask"].any(-1)
        on_policy_mask = ~off_policy_mask
        on_resp_len = response_length[on_policy_mask]
        off_resp_len = response_length[off_policy_mask]
        on_score = sequence_score[on_policy_mask]
        off_score = sequence_score[off_policy_mask]

        if on_resp_len.numel() > 0:
            metrics["on_off_metrics/on_response_length_mean"] = torch.mean(on_resp_len).detach().item()
            metrics["on_off_metrics/on_score"] = torch.mean(on_score).detach().item()
        if off_resp_len.numel() > 0:
            metrics["on_off_metrics/off_response_length_mean"] = torch.mean(off_resp_len).detach().item()
            metrics["on_off_metrics/off_score"] = torch.mean(off_score).detach().item()
        metrics["on_off_metrics/off_on_example_ratio"] = (
            off_policy_mask.sum().item() / max(on_policy_mask.sum().item(), 1)
        )

    return metrics
