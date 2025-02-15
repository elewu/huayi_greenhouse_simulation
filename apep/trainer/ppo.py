import rldev, rllib
# rldev.globv.init()

import collections
import logging
import os
import copy
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


from apep.utils.transformation_tools import global_to_local
from apep.data.format import ModelOutputDim
from apep.model.template import build_module_wrapper
from apep.metric.reward import get_advantage_and_return

# from pep.lightning_module_rl import RolloutBuffer

from apep.trainer.tools import freeze_params, unfreeze_params
from apep.trainer.template import aggregate_objectives
from apep.trainer.il import TrainerIL

logger = logging.getLogger(__name__)







class TrainerPPO(TrainerIL):
    def init_models(self):
        super().init_models()

        ################################################################################################
        ### model old
        self.model_old = copy.deepcopy(self.model)
        freeze_params(self.model_old)
        self.models.update(model_old=self.model_old)

        ################################################################################################
        ### params

        # self.use_critic = self.model.use_critic
        self.rl_opt = self.model.model_params.rl_opt
        self.il_opt = self.model.model_params.il_opt
        self.reward_func = self.model.model_params.reward_func
        # self.weight_anchor = self.model.model_params.weight_anchor
        # self.use_anchor_prob = self.model.model_params.use_anchor_prob
        # self.use_anchor_mix = self.model.model_params.use_anchor_mix

        # self.score_traj = self.model.model_params.score_traj

        self.gamma = self.model.model_params.gamma
        self.lamda = self.model.model_params.lamda
        return


    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.global_step % 8 == 0:
            rllib.utils.hard_update(self.model_old, self.model)
        
        loss = self.step(batch, 'train')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    

    def training_end(self):
        super().training_end()

        if self.distributed:
            torch.distributed.barrier()
            for param in self.model.parameters():
                torch.distributed.broadcast(param.data, src=0)  # Broadcast from rank 0 to all others
            for param in self.model_old.parameters():
                torch.distributed.broadcast(param.data, src=0)  # Broadcast from rank 0 to all others
            torch.distributed.barrier()
        return






    def forward(self, batch, prefix: str) -> torch.Tensor:
        
        data, scenarios = batch

        if self.training:
            goal = data.front_goal if random.random() > 0.5 else data.back_goal
        else:
            goal = data.front_goal
        data.update(goal=goal)

        input_data = data

        # targets = rldev.Data(
        #     trajectory=rldev.Data(data=input_data.ego_gt_trajectory, mask=input_data.ego_gt_trajectory_mask),
        #     object_trajectories=rldev.Data(data=input_data.object_gt_trajectories, mask=input_data.object_gt_trajectories_mask),
        #     object_trajectories_local=rldev.Data(data=input_data.object_gt_trajectories_local, mask=input_data.object_gt_trajectories_mask),
        # )
        targets = None
        
        ### collect data (model_old)
        experience = self.rollout(input_data, scenarios, targets)

        ### genrate trajectory (model)
        with torch.no_grad():
            output_data = self.model(input_data, scenarios)

        objectives = self.unwrapped_model.compute_objectives(input_data, output_data, experience, targets, scenarios, self.global_step)
        # print(rldev.Data(**objectives).keys())
        # print(rldev.Data(**objectives).mean())

        with torch.no_grad():
            metrics = self.compute_metrics(input_data, output_data, targets)
            timewise_metrics = self.compute_timewise_metrics(input_data, output_data, targets)
            benchmark_metrics = self.compute_quality_metrics(input_data, output_data, targets)
            # benchmark_metrics.collision *= output_data.ego_probability[..., None]
            # benchmark_metrics.out_map_boundary *= output_data.ego_probability[..., None]

            benchmark_metrics_new = rldev.Data()
            benchmark_metrics_min = rldev.Data()
            benchmark_metrics_max = rldev.Data()
            for key, value in benchmark_metrics.items():
                value_new = torch.masked.MaskedTensor(value, output_data.trajectory.mask).sum(dim=-1).mean(dim=-1)
                benchmark_metrics_new.update(**{key: torch.where(value_new.get_mask(), value_new.get_data(), 0.0)})
                idx_min = torch.where(output_data.trajectory.mask, value, 1e6).sum(dim=-1).argmin(dim=-1, keepdim=True)
                idx_max = torch.where(output_data.trajectory.mask, value, -1e6).sum(dim=-1).argmax(dim=-1, keepdim=True)

                value_min = value.gather(-2, idx_min[..., None].repeat_interleave(value.shape[-1], dim=-1))
                value_max = value.gather(-2, idx_max[..., None].repeat_interleave(value.shape[-1], dim=-1))
                benchmark_metrics_min.update(**{key+'_min': value_min.sum(dim=-1).mean(dim=-1)})
                benchmark_metrics_max.update(**{key+'_max': value_max.sum(dim=-1).mean(dim=-1)})
            benchmark_metrics = benchmark_metrics_new + benchmark_metrics_min + benchmark_metrics_max
            
            rewards = self.get_reward(input_data, output_data, targets)
            rewards = {f'reward_{key}': value for key, value in rewards.items()}
            metrics.update(**rldev.Data(**rewards).sum(dim=-1).mean(dim=-1).to_dict())
            reward = torch.stack(list(rewards.values()), dim=-1).sum(dim=-1)
            metrics.update(reward=reward.sum(dim=-1).mean(dim=-1))
            
        return output_data, objectives, metrics, timewise_metrics, benchmark_metrics





    @torch.no_grad()
    def rollout(self, data: rldev.Data, scenarios, targets, rl_opt=None):
        output_data = self.model_old(data, scenarios, rl_opt=rl_opt)
        
        states = output_data['states'][:-1]
        actions_data = output_data['actions_data']
        next_states = output_data['states'][1:]
        rewards = torch.stack(list(self.get_reward(data, output_data, targets).values()), dim=-1).sum(dim=-1)
        values_old = torch.stack([action_data.value for action_data in actions_data], dim=-2).squeeze(-1)
        dones = torch.zeros_like(rewards)
        dones[..., -1] = 1
        # dones = (dones.bool() | output_data['benchmark_metrics'].done.bool()).float()

        advantage, returns = get_advantage_and_return(values_old, rewards, dones, self.gamma, self.lamda)

        state = rldev.stack_data(states).stack(dim=1)
        action_data = rldev.stack_data(actions_data).stack(dim=1)
        next_state = rldev.stack_data(next_states).stack(dim=1)
        reward = torch.stack(rewards.split(1, dim=-1), dim=1)
        done = torch.stack(dones.split(1, dim=-1), dim=1)
        advantage = torch.stack(advantage.split(1, dim=-1), dim=1)
        returns = torch.stack(returns.split(1, dim=-1), dim=1)

        experience = rldev.Data(
            state=state, action_data=action_data, next_state=next_state,
            reward=reward, done=done,
            advantage=advantage, returns=returns,
        )
        return experience



    def get_reward(self, input_data, output_data, targets):
        quality_metrics = self.compute_quality_metrics(input_data, output_data, targets)

        reward_funcs = self.reward_func.split('_')
        rewards = {}
        for reward_func in reward_funcs:
            if reward_func == 'invalid':
                raise ValueError(f'Unknown reward function: {reward_func}')
            

            elif reward_func == 'prog':
                reward = quality_metrics.progress_towards_goal
            elif reward_func == 'fap':
                reward = quality_metrics.final_approaches_goal *0.1
            elif reward_func == 'bound':
                reward = quality_metrics.in_bound *10

            else:
                raise ValueError(f'Unknown reward function: {reward_func}')
            # rewards.append(reward)
            rewards[reward_func] = reward
        
        return rewards



