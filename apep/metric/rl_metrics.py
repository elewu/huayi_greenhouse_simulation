import torch

from apep.metric.tools import AbstractMetric




class PPORatio(AbstractMetric):
    def __init__(self, name='ppo_ratio') -> None:
        super().__init__(name=name)

    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        batch_size = output_data.trajectory.data.shape[0]
        action_data = output_data.action_data

        if 'ratio' in action_data.keys():
            ratio = action_data.ratio.mean(dim=-1).mean(dim=-1)
        else:
            batch_size = output_data.trajectory.data.shape[0]
            ratio = torch.ones((batch_size, 1))
        return ratio.mean(dim=1)



class PPORatioRange(AbstractMetric):
    def __init__(self, name='ppo_ratio_range') -> None:
        super().__init__(name=name)

    def compute(self, input_data, output_data, targets) -> torch.Tensor:
        action_data = output_data.action_data

        if 'ratio' in action_data.keys():
            ratio = action_data.ratio.mean(dim=-1).mean(dim=-1)
        else:
            batch_size = output_data.trajectory.data.shape[0]
            ratio = torch.ones((batch_size, 1))
        range = (ratio - 1.0).abs()
        return range.mean(dim=1)




