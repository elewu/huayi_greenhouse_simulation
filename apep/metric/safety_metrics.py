import rldev

import copy
from hydra.utils import instantiate

import torch
import torch.nn.functional as F

from apep.data.format import ModelOutputDim, ObjectTensorOutputSize
from apep.metric.tools import AbstractMetric




class ProgressTowardsGoal(AbstractMetric):
    def __init__(self, cfg, name='progress_towards_goal') -> None:
        super().__init__(name=name)

        # self.vehicle_parameters = instantiate(cfg.ego_vehicle_parameters)

        self._max_violation_threshold = 0.3


    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        B, M, T, _ = output_data.trajectory.data.shape

        goal = input_data.goal[..., None,None, :]
        pose = torch.cat([input_data.pose[..., None,None, :].repeat_interleave(M, dim=ObjectTensorOutputSize.mode), output_data.trajectory.data], dim=-2)

        return -(pose - goal)[..., :2].norm(dim=-1).diff(dim=-1)


class FinalApproachesGoal(AbstractMetric):
    def __init__(self, cfg, name='final_approaches_goal') -> None:
        super().__init__(name=name)

    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        B, M, T, _ = output_data.trajectory.data.shape
        goal = input_data.goal[..., None, :]
        final_pose = output_data.trajectory.data[..., -1, :]
        dist = (final_pose - goal)[..., :2].norm(dim=-1)
        # dist = (final_pose - goal).norm(dim=-1)
        zeros = torch.zeros_like(dist).unsqueeze(-1).repeat_interleave(T-1, dim=-1)
        return torch.cat([zeros, -dist.unsqueeze(-1)], dim=-1)



class InBound(AbstractMetric):
    def __init__(self, cfg, name='in_bound', interpolate_scale=1, pad=False) -> None:
        super().__init__(name=name)

        self.interpolate_scale = interpolate_scale
        self.pad = pad
    
    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        # aussume map is horizontal

        bound = input_data.map_polylines[..., 0,1].sort(dim=-1).values
        bound_min, bound_max = bound[..., 0][..., None,None], bound[..., 1][..., None,None]

        y = output_data.trajectory.data[..., ModelOutputDim.y]
        in_bound = (y > bound_min) & (y < bound_max)
        return in_bound.to(torch.float32) -1


class NearCenter(AbstractMetric):
    def __init__(self, cfg, name='near_center', interpolate_scale=1, pad=False) -> None:
        super().__init__(name=name)

        self.interpolate_scale = interpolate_scale
        self.pad = pad
    
    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        # aussume map is horizontal

        bound = input_data.map_polylines[..., 0,1].mean(dim=-1)[..., None,None]
        y = output_data.trajectory.data[..., ModelOutputDim.y]
        return -(y - bound).abs()

