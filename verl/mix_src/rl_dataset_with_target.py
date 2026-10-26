"""
RLHF Dataset with teacher target traces for LUFFY off-policy learning.
Adapted from LUFFY to work with the current verl RLHFDataset API.

The new RLHFDataset takes (data_files, tokenizer, config, processor) so we
cannot simply subclass it with old-style keyword args. Instead we subclass it
properly and override __getitem__ to also emit `tgt_input_ids`.
"""

import copy
import logging
import os
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length

logger = logging.getLogger(__name__)


class RLHFDatasetWithTarget(RLHFDataset):
    """Extends RLHFDataset to also load teacher target traces.

    Extra config keys consumed (via config DictConfig):
        - target_key (str): column name for teacher target, default "target"
        - max_target_length (int): max tokens for target, default 8192
        - filter_targets (bool): whether to filter empty targets, default False
        - sample_target_ratio (float): probability of using the target, default 1.0
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)

        self.target_key = config.get("target_key", "target")
        self.max_target_length = config.get("max_target_length", 8192)
        self.filter_targets = config.get("filter_targets", False)
        self.sample_target_ratio = config.get("sample_target_ratio", 1.0)

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)

        tgt = self.dataframe[item].get(self.target_key, None)
        sample = np.random.rand() < self.sample_target_ratio
        # breakpoint()
        raw_prompt = None
        if self.processor is not None:
            messages = copy.deepcopy(self.dataframe[item].get(self.config.get("prompt_key", "prompt"), []))
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
        else:
            messages = copy.deepcopy(self.dataframe[item].get(self.config.get("prompt_key", "prompt"), []))
            if self.config.get("prompt_type", None) is None:
                raw_prompt = messages[0]["content"] if messages else ""
            else:
                raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)

        # breakpoint()
        if tgt is not None and sample:
            if isinstance(tgt, list):
                tgt = tgt[0]

            tgt_content = tgt["content"] if isinstance(tgt, dict) else str(tgt)

            if raw_prompt and raw_prompt.endswith("<think>\n") and tgt_content.startswith("<think>\n"):
                tgt_content = tgt_content[len("<think>\n"):]

            tgt_input_ids = self.tokenizer(
                tgt_content, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].reshape(-1)
            tgt_input_ids = tgt_input_ids.reshape(1, -1)
        else:
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0)

        # breakpoint()
        seq_len = tgt_input_ids.shape[-1]
        if seq_len < self.max_target_length:
            tgt_input_ids = pad_sequence_to_length(
                tgt_input_ids,
                max_seq_len=self.max_target_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=False,
            )
        else:
            tgt_input_ids = tgt_input_ids[:, : self.max_target_length]

        # breakpoint()
        row_dict["tgt_input_ids"] = tgt_input_ids.squeeze(0)
        return row_dict


class ResumableRandomSampler(torch.utils.data.Sampler):
    """Deterministic random sampler that supports checkpoint/resume."""

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1].item()

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {
            "perm": self.perm,
            "perm_index": self.perm_index,
            "generator_state": self.generator.get_state(),
        }

    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])
