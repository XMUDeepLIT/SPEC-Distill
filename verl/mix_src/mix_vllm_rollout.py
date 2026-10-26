"""
LUFFY vLLM rollout with teacher prefix injection.
Adapted from LUFFY to work with the current verl vLLMRollout API.

Key changes from original LUFFY:
  - vLLMRollout.__init__ now takes model_path (str) instead of actor_module
  - generate_sequences uses vllm_inputs format and pad_2d_list_to_length
  - get_eos_mask -> get_response_mask
  - Integrates with current verl sharding / profiler infrastructure
"""

import logging
import os
import random
from copy import deepcopy
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import (
    get_response_mask,
    pad_2d_list_to_length,
    pad_sequence_to_length,
)
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    token_ids = prompt_token_ids[: non_pad_index[-1][0] + 1].tolist()
    return token_ids


class MIXvLLMRollout(vLLMRollout):
    """vLLM rollout that injects teacher-trace prefix into prompts.

    During training, for each prompt that has a `tgt_input_ids`, a random
    fraction of the teacher trace is prepended to the prompt.  The generated
    response is then the concatenation of this prefix and the model's own
    continuation.  A `prefix_mask` tensor marks which response tokens came
    from the teacher.
    """

    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs.get("tokenizer", None)
        super().__init__(*args, **kwargs)

        self.prefix_strategy = self.config.get("prefix_strategy", "random")
        self.prefix_steps = self.config.get("prefix_steps", 300)
        self.prefix_linear_max_ratio = self.config.get("prefix_linear_max_ratio", 0.8)

        if self.prefix_strategy == "linear_max":
            self.prefix_ratio_windows = [
                (0, i * self.prefix_linear_max_ratio / 10) for i in range(10, 0, -1)
            ]
            self.prefix_step_windows = [
                (i * self.prefix_steps / 10, (i + 1) * self.prefix_steps / 10) for i in range(10)
            ]
        elif self.prefix_strategy == "reverse_linear":
            self.prefix_ratio_windows = [
                (0, (i + 1) * self.prefix_linear_max_ratio / 10) for i in range(10)
            ]
            self.prefix_step_windows = [
                (i * self.prefix_steps / 10, (i + 1) * self.prefix_steps / 10) for i in range(10)
            ]
        elif self.prefix_strategy == "fixed":
            n_prefix = self.config.n_prefix if self.config.n_prefix != -1 else self.config.n
            ratio_step = (self.config.max_prefix_ratio - self.config.min_prefix_ratio) / max(n_prefix - 1, 1)
            self.prefix_fix_ratios = [
                self.config.min_prefix_ratio + i * ratio_step for i in range(n_prefix)
            ]

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        idx_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]

        do_sample = prompts.meta_info.get("do_sample", True)
        if do_sample:
            idx_list = sum([[idx_list[i]] * self.config.n for i in range(len(idx_list))], [])

        prefix_ratios = None
        tgt_input_ids = None
        prefix_list = None

        if "tgt_input_ids" in prompts.batch:
            tgt_input_ids = prompts.batch["tgt_input_ids"]
            tgt_list = [
                _pre_process_inputs_right_pad(self.pad_token_id, tgt_input_ids[i])
                for i in range(batch_size)
            ]
            tgt_list = [
                tgt_list[i] + [self.tokenizer.eos_token_id] if len(tgt_list[i]) > 0 else tgt_list[i]
                for i in range(batch_size)
            ]
            tgt_list = sum([[tgt_list[i]] * self.config.n for i in range(len(tgt_list))], [])

            global_steps = prompts.meta_info.get("global_steps", 1) - 1

            prefix_ratios = self._compute_prefix_ratios(
                batch_size, len(tgt_list), global_steps
            )
            prefix_list = [
                tgt_list[i][: int(len(tgt_list[i]) * prefix_ratios[i])] for i in range(len(tgt_list))
            ]
            idx_list = [idx_list[i] + prefix_list[i] for i in range(len(idx_list))]
        else:
            tgt_list = None

        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        is_validate = prompts.meta_info.get("validate", False)
        if is_validate:
            kwargs["top_k"] = self.config.val_kwargs.top_k
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature
            kwargs["n"] = 1

        if prompts.meta_info.get("val_temperature", None):
            kwargs["temperature"] = prompts.meta_info["val_temperature"]

        kwargs["n"] = 1

        vllm_inputs = [{"prompt_token_ids": ids} for ids in idx_list]
        for inp in vllm_inputs:
            if isinstance(inp["prompt_token_ids"], np.ndarray):
                inp["prompt_token_ids"] = inp["prompt_token_ids"].tolist()

        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=True,
            )

        response_ids_list = []
        for output in outputs:
            for sample in output.outputs:
                response_ids_list.append(list(sample.token_ids))

        if "tgt_input_ids" in prompts.batch and prefix_list is not None:
            concat_resp_list = []
            prefix_mask = torch.zeros(
                [len(response_ids_list), self.config.response_length], dtype=torch.bool
            ).to(idx.device)

            for i in range(len(response_ids_list)):
                full_resp = prefix_list[i] + response_ids_list[i]
                concat_resp_list.append(full_resp)
                prefix_len = min(len(prefix_list[i]), self.config.response_length)
                prefix_mask[i, :prefix_len] = True

            response = pad_2d_list_to_length(
                concat_resp_list, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
        else:
            response = pad_2d_list_to_length(
                response_ids_list, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            prefix_mask = torch.tensor([])

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
        elif response.shape[1] > self.config.response_length:
            response = response[:, : self.config.response_length]

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            if tgt_input_ids is not None:
                tgt_input_ids = tgt_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if tgt_input_ids is not None:
            batch["tgt_input_ids"] = tgt_input_ids
        if prefix_mask.shape[0] > 0:
            batch["prefix_mask"] = prefix_mask

        meta_info = {}
        if prefix_ratios is not None:
            meta_info["prefix_ratios"] = prefix_ratios
        return DataProto(batch=batch, meta_info=meta_info)

    def _compute_prefix_ratios(self, batch_size, total_samples, global_steps):
        """Compute prefix ratios based on the configured strategy."""
        prefix_share = self.config.get("prefix_share_across_samples", True)
        n_prefix_cfg = self.config.get("n_prefix", -1)
        n = self.config.n

        if not prefix_share:
            if n_prefix_cfg == -1:
                return self._ratios_for_strategy(total_samples, global_steps)
            else:
                assert n_prefix_cfg <= n
                ratios = []
                for _ in range(batch_size):
                    ratios.extend(self._ratios_for_strategy(n_prefix_cfg, global_steps))
                    ratios.extend([0.0] * (n - n_prefix_cfg))
                return ratios
        else:
            if self.prefix_strategy == "linear":
                ratio = min(global_steps / self.prefix_steps, 1.0)
                base = self.prefix_linear_max_ratio * (1 - ratio)
            else:
                base = None

            assert n_prefix_cfg <= n or n_prefix_cfg == -1
            ratios = []
            for _ in range(batch_size):
                r = base if base is not None else random.uniform(
                    self.config.min_prefix_ratio, self.config.max_prefix_ratio
                )
                if n_prefix_cfg > 0:
                    ratios.extend([r] * n_prefix_cfg)
                    ratios.extend([0.0] * (n - n_prefix_cfg))
                else:
                    ratios.extend([r] * n)
            return ratios

    def _ratios_for_strategy(self, count, global_steps):
        if self.prefix_strategy == "random":
            return [
                random.uniform(self.config.min_prefix_ratio, self.config.max_prefix_ratio)
                for _ in range(count)
            ]
        elif self.prefix_strategy in ("reverse_linear", "linear_max"):
            w_idx = 0
            for i in range(len(self.prefix_step_windows)):
                if self.prefix_step_windows[i][0] <= global_steps <= self.prefix_step_windows[i][1]:
                    w_idx = i
                    break
            return [
                random.uniform(*self.prefix_ratio_windows[w_idx]) for _ in range(count)
            ]
        elif self.prefix_strategy == "fixed":
            return list(self.prefix_fix_ratios[:count])
        else:
            raise NotImplementedError(f"Prefix strategy {self.prefix_strategy}")
