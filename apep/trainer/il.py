import rldev
# rldev.globv.init()

import os
import logging
import copy
import tqdm

import hydra
from typing import Dict

import numpy as np
import torch

from apep.utils.trajectory_sampling import TrajectorySampling
from apep.dataset.tools import get_trajectory_sampling_slice

from .template import TrainerTemplate

logger = logging.getLogger(__name__)



class TrainerIL(TrainerTemplate):
    def init_models(self):
        super().init_models()

        ################################################################################################
        ### params
        # self.max_agents = self.cfg.model.feature_params.max_agents

        self.past_trajectory_sampling = hydra.utils.instantiate(self.cfg.model.feature_params.past_trajectory_sampling)
        self.future_trajectory_sampling = hydra.utils.instantiate(self.cfg.model.target_params.future_trajectory_sampling)
        max_past_trajectory_sampling = TrajectorySampling(time_horizon=self.cfg.max_history_time, interval_length=0.1)
        max_future_trajectory_sampling = TrajectorySampling(time_horizon=8.0, interval_length=0.1)

        past_ts = get_trajectory_sampling_slice(self.past_trajectory_sampling, max_past_trajectory_sampling)
        past_ts = np.concatenate([-past_ts[::-1] -2, np.array([-1])])
        future_ts = get_trajectory_sampling_slice(self.future_trajectory_sampling, max_future_trajectory_sampling)
        self.past_ts = past_ts
        self.future_ts = future_ts
        return




    def forward(self, batch, prefix: str):
        data, scenarios = batch

        input_data = copy.copy(data)

        targets = rldev.Data(
            trajectory=rldev.Data(data=input_data.ego_gt_trajectory, mask=input_data.ego_gt_trajectory_mask),
            object_trajectories=rldev.Data(data=input_data.object_gt_trajectories, mask=input_data.object_gt_trajectories_mask),
            object_trajectories_local=rldev.Data(data=input_data.object_gt_trajectories_local, mask=input_data.object_gt_trajectories_mask),
        )

        if self.mix_train and prefix != 'train':
        # if False:
            # data.pop('ego_gt_trajectory')
            # data.pop('ego_gt_trajectory_mask')
            input_data.pop('object_gt_trajectories')
            input_data.pop('object_gt_trajectories_mask')

        output_data = self.forward_model(input_data, scenarios)

        objectives = self.unwrapped_model.compute_objectives(input_data, output_data, targets, scenarios)
        #     objectives = {'pseudo': torch.zeros(data.obj_trajs.shape[0])}
        with torch.no_grad():
            metrics = self.compute_metrics(input_data, output_data, targets)
            timewise_metrics = self.compute_timewise_metrics(input_data, output_data, targets)
            benchmark_metrics = self.compute_quality_metrics(input_data, output_data, targets)
        # benchmark_metrics = output_data.benchmark_metrics
        # print(rldev.Data(**objectives).keys())
        # print(rldev.Data(**objectives).mean())

        return output_data, objectives, metrics, timewise_metrics, benchmark_metrics



    def compute_timewise_metrics(self, input_data, output_data, targets) -> Dict[str, torch.Tensor]:
        return rldev.Data(**{metric_name: metric.compute(input_data, output_data, targets) for metric_name, metric in self.unwrapped_model.timewise_metrics.items() if metric.training == self.training})

    def compute_quality_metrics(self, input_data, output_data, targets) -> Dict[str, torch.Tensor]:
        return rldev.Data(**{metric_name: metric.compute(input_data, output_data, targets) for metric_name, metric in self.unwrapped_model.quality_metrics.items()})




