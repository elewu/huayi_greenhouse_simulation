
from typing import Dict, List, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from apep.objective.tools import AbstractObjective



######################################################################################################
### ppo ##############################################################################################
######################################################################################################



class PPOPolicyObjective(AbstractObjective):
    def __init__(self, name='ppo_policy_objective', weight=100.0):
        super().__init__(name=name ,weight=weight)

        self.epsilon_clip = 0.2


    def compute(self, action_data, experience, scenarios=None, mode_idxs=None) -> torch.Tensor:
        # batch_size = experience.advantage.shape[0]

        logprob_old = experience.action_data.logprob
        advantage = experience.advantage

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        ratio = (action_data.logprob - logprob_old).exp()
        action_data.update(ratio=ratio)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage
        loss = -torch.min(surr1, surr2)
        # loss = torch.where((1-experience.done).bool(), loss, 0.0)

        loss = loss.transpose(1, 2)
        loss = loss.mean(dim=-1).mean(dim=-1)
        if mode_idxs != None:
            loss = loss.gather(1, mode_idxs)
            # loss = loss[torch.arange(batch_size)[..., None], mode_idxs]
        return self.weight * loss.mean(dim=1)





class PPOEntropyObjective(AbstractObjective):
    def __init__(self, name='ppo_entropy_objective', weight=0.001):
        super().__init__(name=name ,weight=weight)

    def compute(self, action_data, experience, scenarios=None, mode_idxs=None) -> torch.Tensor:
        # batch_size = action_data.mu.shape[0]

        loss = - MultivariateNormal(action_data.mu, torch.diag_embed(action_data.var)).entropy().mean(dim=1)
        if mode_idxs != None:
            loss = loss.gather(1, mode_idxs)
            # loss = loss[torch.arange(batch_size)[..., None], mode_idxs]
        return self.weight * loss.mean(dim=1)







class PPOValueObjective(AbstractObjective):
    def __init__(self, name='ppo_value_objective', weight=0.5, clip=False):
        super().__init__(name=name ,weight=weight)
        self.clip = clip
        self.clip_range = 0.2

        self.loss_func = torch.nn.modules.loss.MSELoss(reduction='none')


    def compute(self, action_data, experience, scenarios=None, mode_idxs=None) -> torch.Tensor:
        returns = experience.returns
        value_old = experience.action_data.value

        loss = self.loss_func(action_data.value, returns).mean(dim=-1).transpose(1,2).mean(dim=-1)
        if self.clip:
            raise NotImplementedError
            v_clipped = value_old + torch.clamp(action_data.value - value_old, -self.clip_range, self.clip_range)
            loss_clipped = self.loss_func(v_clipped, returns).mean(dim=-1).mean(dim=-1)
            loss = torch.max(loss, loss_clipped)
        
        if mode_idxs != None:
            loss = loss.gather(1, mode_idxs)
        return self.weight * loss.mean(dim=1)


