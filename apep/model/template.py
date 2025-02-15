import rldev

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, cast
import os
import numpy as np
from hydra.utils import instantiate

import torch
import torch.nn as nn


from apep.utils.trajectory_sampling import TrajectorySampling


from apep.utils.transformation_tools import global_to_local, local_to_global
from apep.data.format import ModelOutputDim
# from pep.utils.feature_builder_vector_agents import VectorAgentsFeatureBuilder
# from pep.utils.feature_builder_vector_hdmap import VectorHDMapFeatureBuilder
# from pep.utils.target_builder_ego_trajectory import EgoTrajectoryTargetBuilder
# from pep.utils.target_builder_object_trajectories import ObjectTrajectoriesTargetBuilder

from apep.objective.tools import AbstractObjective
from apep.metric.tools import AbstractMetric



@dataclass
class MotionPlanningTransformerParams:
    context_encoder: Dict

    control_space: bool
    # scenario_builder: NuPlanScenarioBuilder
    max_action: float



@dataclass
class MotionPlanningTransformerFeatureParams:
    # feature_types: Dict[str, int]
    total_max_points: int
    feature_dimension: int
    agent_features: List[str]
    ego_dimension: int
    agent_dimension: int
    max_agents: int
    past_trajectory_sampling: TrajectorySampling
    map_features: List[str]
    max_elements: Dict[str, int]
    max_points: int
    vector_set_map_feature_radius: int
    interpolation_method: str
    disable_map: bool
    disable_agents: bool


@dataclass
class MotionPlanningTransformerTargetParams:
    # num_output_features: int
    future_trajectory_sampling: TrajectorySampling






class ModelWrapper(nn.Module):
    def __init__(
        self,
        model_params: rldev.BaseData,
        feature_params: MotionPlanningTransformerFeatureParams,
        target_params: MotionPlanningTransformerTargetParams,
        meta_cfg: rldev.BaseData,
        objectives: List[AbstractObjective],
        metrics: List[AbstractMetric],
        timewise_metrics = [],
        quality_metrics: List[AbstractMetric] = [],
    ):
        super().__init__()
        self.model_params = model_params
        self.feature_params = feature_params
        self.target_params = target_params

        self.meta_cfg = meta_cfg
        self.cfg = meta_cfg.cfg
        self.objectives = {objective.name: objective for objective in objectives}
        self.metrics = {metric.name: metric for metric in metrics}
        self.timewise_metrics = {metric.name: metric for metric in timewise_metrics}
        self.quality_metrics = {metric.name: metric for metric in quality_metrics}

        # self.future_trajectory_sampling = target_params.future_trajectory_sampling
        return

    def compute_objectives(self, input_data, output_data: rldev.Data, targets: rldev.Data, scenarios) -> Dict[str, torch.Tensor]:
        return {objective.name: objective.compute(input_data, output_data, targets, scenarios) for objective in self.objectives.values()}





def compute_objectives(objectives: List[AbstractObjective], *args, **kwargs) -> Dict[str, torch.Tensor]:
    return {objective.name: objective.compute(*args, **kwargs) for objective in objectives}





def build_module_wrapper(cfg: DictConfig, meta_cfg: DictConfig) -> ModelWrapper:
    # OmegaConf.set_struct(cfg, False)
    # # cfg.pop('objective')
    # # cfg.pop('training_metric')
    # cfg.pop('scenario_type_weights')
    # OmegaConf.set_struct(cfg, True)
    model = instantiate(cfg, meta_cfg=rldev.Data(cfg=meta_cfg))

    return model








class IdentityModel(nn.Module):
    StateDim = ModelOutputDim
    ActionDim = ModelOutputDim
    # class ActionDim:
    #     x = 0
    #     y = 1
    #     heading = 2
    #     dim = 3

    def __init__(self, vehicle_parameters, future_trajectory_sampling: TrajectorySampling):
        super().__init__()
        return


    def forward_old(self, state, action):
        ego_pose = state[..., [ModelOutputDim.x, ModelOutputDim.y, ModelOutputDim.heading]]
        ego_v = state[..., [ModelOutputDim.v]]

        global_pose = local_to_global(action[..., [ModelOutputDim.x, ModelOutputDim.y, ModelOutputDim.heading]], ego_pose)
        next_state = torch.cat([global_pose, action[..., [ModelOutputDim.v]] + ego_v], dim=-1)
        return next_state

    def forward(self, state, action):
        return action


