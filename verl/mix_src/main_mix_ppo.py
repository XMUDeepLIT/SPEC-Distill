"""
LUFFY main entry point.  Adapted from the original LUFFY main_mix_ppo.py to
work with the current verl codebase.

Usage:
    python -m verl.mix_src.main_mix_ppo  <hydra overrides...>

It mirrors the standard verl main_ppo but:
  - Uses MIXActorRolloutRefWorker (prefix-injection rollout + on/off-policy actor)
  - Uses MIXRayPPOTrainer (grpo_split advantage, alpha/beta weighting)
  - Uses RLHFDatasetWithTarget (teacher targets in the dataset)
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available


@hydra.main(config_path="../trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_luffy(config)


def run_luffy(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )
    # breakpoint()
    runner = LUFFYTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class LUFFYTaskRunner:
    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer, hf_processor

        print(f"LUFFYTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Worker setup: use MIXActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import CriticWorker
        from verl.mix_src.mix_fsdp_worker import MIXActorRolloutRefWorker
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        ray_worker_group_cls = RayWorkerGroup

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(MIXActorRolloutRefWorker),
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # Critic (only for GAE)
        if config.algorithm.adv_estimator == "gae":
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
            mapping[Role.Critic] = global_pool_id

        # Reference policy
        if config.algorithm.get("use_kl_in_reward", False) or config.actor_rollout_ref.actor.get("use_kl_loss", False):
            role_worker_mapping[Role.RefPolicy] = ray.remote(MIXActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Reward model
        if config.reward_model.enable:
            from verl.workers.fsdp_workers import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.mix_src.mix_trainer import MIXRayPPOTrainer

        trainer = MIXRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
