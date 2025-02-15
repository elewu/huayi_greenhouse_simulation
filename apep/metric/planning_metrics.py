import rldev

import torch

from apep.data.format import ModelOutputDim
from apep.metric.tools import AbstractMetric





class MeanAcceleration(AbstractMetric):
    def __init__(self, name ='meanAcceleration') -> None:
        super().__init__(name=name)


    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        predicted_trajectory = output_data.trajectory
        if predicted_trajectory.data.shape[-2] <= 2:
            return torch.zeros_like(predicted_trajectory.data[..., 0,0,0])

        t = predicted_trajectory.time_stamps
        dt = t.diff(dim=-1).mean(dim=-1).unsqueeze(-1)

        x = predicted_trajectory.data[..., ModelOutputDim.x]
        y = predicted_trajectory.data[..., ModelOutputDim.y]

        vx = x.diff(dim=-1) /dt
        vy = y.diff(dim=-1) /dt

        ax = vx.diff(dim=-1) /dt
        ay = vy.diff(dim=-1) /dt

        traj_acc = torch.hypot(ax, ay).mean(dim=-1)
        return traj_acc.mean(dim=1)




class MeanJerk(AbstractMetric):
    def __init__(self, name='meanJerk') -> None:
        super().__init__(name=name)


    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        predicted_trajectory = output_data.trajectory
        if predicted_trajectory.data.shape[-2] <= 3:
            return torch.zeros_like(predicted_trajectory.data[..., 0,0,0])
        
        t = predicted_trajectory.time_stamps
        dt = t.diff(dim=-1).mean(dim=-1).unsqueeze(-1)

        x = predicted_trajectory.data[..., ModelOutputDim.x]
        y = predicted_trajectory.data[..., ModelOutputDim.y]

        vx = x.diff(dim=-1) /dt
        vy = y.diff(dim=-1) /dt

        ax = vx.diff(dim=-1) /dt
        ay = vy.diff(dim=-1) /dt

        jx = ax.diff(dim=-1) /dt
        jy = ay.diff(dim=-1) /dt

        traj_jerk = torch.hypot(jx, jy).mean(dim=-1)        
        return traj_jerk.mean(dim=1)







class TrajectoryVariance(AbstractMetric):
    def __init__(self, name='trajectory_variance') -> None:
        super().__init__(name=name)


    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        action_data = rldev.stack_data(output_data['actions_data']).stack(dim=2)
        return action_data.var.sum(dim=-1).mean(dim=-1).mean(dim=1)


