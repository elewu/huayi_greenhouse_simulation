import rldev

import random
from typing import Any, List, Optional
import copy

import os
import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data
import torch.utils.data.distributed

# from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

from apep.dataset.builder import transfer_batch_to_device
from apep.visualize.fig_class import FigClass
import apep.visualize.vis as vis


class VisCallback:
    """
    Callback that visualizes planner model inputs/outputs and logs them in Tensorboard.
    """

    def __init__(
        self, dataset_name, output_dir: str,
        vis_mode: str,
        # vehicle_parameters,
        images_per_tile: int,
        num_train_tiles: int,
        num_val_tiles: int,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        if self.dataset_name in ['greenhouse']:
            pass
        else:
            raise NotImplementedError
        self.output_dir = output_dir
        self.vis_mode = vis_mode
        self.extension = 'png'
        self.save_dir = os.path.join(output_dir, f'vis_{vis_mode}_callback')
        os.makedirs(self.save_dir, exist_ok=True)
        # self.vehicle_parameters = vehicle_parameters
        self.custom_batch_size = min(images_per_tile, 8)
        self.num_train_samples = num_train_tiles * self.custom_batch_size
        self.num_val_samples = num_val_tiles * self.custom_batch_size

        self.dataloaders = {}



    def create_dataloader(self, dataset, num_samples: int, idxs=None) -> torch.utils.data.DataLoader:
        dataset = copy.deepcopy(dataset)
        if idxs != None:
            dataset.set_subset(idxs=idxs)
        else:
            dataset_size = len(dataset)
            num_keep = min(dataset_size, num_samples)
            ratio = num_keep / dataset_size
            dataset.set_subset(fraction=ratio)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.custom_batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            num_workers=16,
            pin_memory=True,
        )




    @torch.no_grad()
    def on_validation_start(self, trainer, idxs=None):
        # for prefix in trainer.val_data_types:
        if True:
            prefix = trainer.val_data_types[0]
            os.makedirs(os.path.join(self.save_dir, prefix), exist_ok=True)

            dataset = trainer.validation_datasets[prefix]
            self.dataloaders[prefix] = self.create_dataloader(dataset, self.num_val_samples, idxs=idxs)
        return

    @torch.no_grad()
    def on_validation_end(self, trainer):
        for prefix, dataloader in self.dataloaders.items():
            self._log_from_dataloader(
                trainer,
                dataloader,
                trainer.epoch_idx,
                prefix,
            )
        return



    def _log_from_dataloader(
        self,
        trainer,
        dataloader: torch.utils.data.DataLoader,
        current_epoch: int,
        prefix: str,
    ) -> None:
        for batch_idx, batch in enumerate(dataloader):
            batch = transfer_batch_to_device(batch, trainer.device)
            data, scenarios = batch

            input_data = data
            targets = None

            output_data, objectives, metrics, _, benchmark_metrics = trainer.forward(batch, prefix)
            try:
                output_data.pop('states')
                output_data.pop('actions_data')
            except:
                pass
            metrics = rldev.Data(**metrics)

            self._log_batch(input_data, output_data, targets, metrics, benchmark_metrics, scenarios, batch_idx, current_epoch, prefix)



    def _log_batch(
        self,
        input_data: rldev.Data,
        output_data: rldev.Data,
        targets: rldev.Data,
        metrics,
        benchmark_metrics,
        scenarios,
        batch_idx: int,
        current_epoch: int,
        prefix: str,
    ) -> None:
        ### organize data
        try:
            output_data.pop('benchmark_metrics')
        except:
            pass
        trajectory_score = output_data.get('ego_probability', None)
        q_value = output_data.get('q_value', None)
        reward = output_data.get('reward', None)
        # output_data.pop('open_state_dropout')

        # input_data += targets

        ### vis
        batch_size = input_data.pose.shape[0]
        metrics_numpy = metrics.cpu().numpy()
        benchmark_metrics_numpy = benchmark_metrics.cpu().numpy()
        trajectory_score_numpy = trajectory_score.cpu().numpy() if trajectory_score != None else [None] *batch_size
        q_value_numpy = q_value.cpu().numpy() if q_value != None else [None] *batch_size
        reward_numpy = reward.cpu().numpy() if reward != None else [None] *batch_size


        for data_index in range(batch_size):
            save_path = os.path.join(self.save_dir, prefix, f'{current_epoch}--{scenarios[data_index].data_idx}--{scenarios[data_index].episide_idx}-{scenarios[data_index].time_idx}.{self.extension}')
            vis_func(
                self,
                save_path,
                scenarios[data_index],
                input_data[[data_index]].cpu().numpy(),
                output_data[[data_index]].cpu().numpy(),
                # vehicle_parameters=self.vehicle_parameters,
                others=rldev.Data(
                    metrics=metrics_numpy[data_index],
                    benchmark_metrics=benchmark_metrics_numpy[data_index],
                    trajectory_score=trajectory_score_numpy[data_index],
                    q_value=q_value_numpy[data_index],
                    reward=reward_numpy[data_index],
                ),
            )
        return



def vis_func(self, save_path, scenario, input_data, output_data, others):
    batch_idx = 0

    num_rows = 1
    num_columns = 1
    fig_class = FigClass(num_rows, num_columns, dpi=1000)
    fig_class.resize(5)
    fig_class.align()

    ax = fig_class.axes[0]

    # map
    left_bound, right_boud = input_data.map_polylines[batch_idx]
    ax.plot(left_bound[:, 0], left_bound[:, 1], '-', color='grey', linewidth=1)
    ax.plot(right_boud[:, 0], right_boud[:, 1], '-', color='grey', linewidth=1)

    # goal
    draw_pose(ax, input_data.front_goal[batch_idx], color='red', label='front goal')
    draw_pose(ax, input_data.back_goal[batch_idx], color='purple', label='back goal')

    goal = input_data.goal[batch_idx]
    ax.plot(goal[0], goal[1], 'x', color='black', linewidth=4)

    # current pose
    draw_pose(ax, input_data.pose[batch_idx], color='black', label='current pose')

    # trajectory
    traj = output_data.trajectory.data[batch_idx, 0]
    ax.plot(traj[:, 0], traj[:, 1], '-', color='blue', linewidth=2)
    for p in traj:
        draw_pose(ax, p, color='green')
    
    ax.legend(fontsize=4)

    fig_class.fig.savefig(save_path, bbox_inches='tight')
    fig_class.close()
    return


def draw_pose(ax, pose, length=1, color='black', label=''):
    ax.arrow(pose[0], pose[1], length* np.cos(pose[2]), length* np.sin(pose[2]), head_width=0.1, head_length=0.2, fc=color, ec=color, label=label)

