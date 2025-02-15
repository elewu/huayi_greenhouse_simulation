
from dataclasses import dataclass
import numpy as np





def get_trajectory_sampling_slice(trajectory_sampling, max_trajectory_sampling):
    max_ts = np.arange(0.0, max_trajectory_sampling.time_horizon, max_trajectory_sampling.interval_length) + max_trajectory_sampling.interval_length
    ts = np.arange(0.0, trajectory_sampling.time_horizon, trajectory_sampling.interval_length) + trajectory_sampling.interval_length
    index = np.where(np.isin(max_ts.round(1), ts.round(1)))[0]
    assert index.shape == ts.shape
    return index






@dataclass
class CacheScenario:
    data_idx: int
    episide_idx: int
    time_idx: int

    def get_name(self):
        return f'{self.episide_idx}--{self.time_idx}'

    def add_attribute(self, attr_name: str, value):
        setattr(self, attr_name, value)

