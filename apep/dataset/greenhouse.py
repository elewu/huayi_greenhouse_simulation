import rldev

import pickle
import collections
import logging
import os
import numpy as np
import hydra
import time

import torch
from torch.utils.data import Dataset

from hydra.utils import instantiate
from omegaconf import DictConfig

from .tools import CacheScenario

logger = logging.getLogger(__name__)




def create_dataset(cfg, base_dir, dataset_fraction: float, dataset_name: str, selected_scenarios_types=None, num_scenarios_per_type=-1):
    dataset = GreenhouseDataset(cfg, base_dir, dataset_name)
    if selected_scenarios_types != None:
        dataset.set_subset(selected_scenarios_types=selected_scenarios_types, num_scenarios_per_type=num_scenarios_per_type)
    if dataset_fraction < 1:
        dataset.set_subset(fraction=dataset_fraction)
    # print_scenario_types(dataset.scenarios)
    return dataset




class GreenhouseDataset(Dataset):
    def __init__(self, cfg, base_dir, data_type) -> None:
        super().__init__()
        self.subset = False

        self.cfg = cfg
        self.base_dir = base_dir
        self.data_type = data_type
        self.data_dir = os.path.join(base_dir, data_type)

        # if hasattr(cfg, 'model_prediction'):
        #     self.extra_prediction_dir = os.path.join(base_dir, 'extra_prediction', cfg.model_prediction.split('/')[-1])
        # else:
        #     self.extra_prediction_dir = None

        file_info = os.path.join(base_dir, f'file_info_{data_type}.pkl')
        if os.path.isfile(file_info):
        # if False:
        #     pass
            with open(file_info, 'rb') as fp:
                self._file_names = pickle.load(fp)
        else:
            t1 = time.time()
            _file_names = os.listdir(self.data_dir)
            _file_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[3]))
            _file_names.sort(key=lambda x: int(x.split('_')[1]))
            self._file_names = [os.path.join(self.data_dir, file_name) for file_name in _file_names]
            with open(file_info, 'wb') as fp:
                pickle.dump(self._file_names, fp)
            t2 = time.time()
            logger.info(f'[GreenhouseDataset] online generate file info: {t2-t1} s')

        scenarios = []
        for data_idx, file_name in enumerate(self._file_names):
            _, episide_idx, _, time_idx = os.path.splitext(os.path.basename(file_name))[0].split('_')
            scenarios.append(CacheScenario(data_idx, int(episide_idx), int(time_idx)))
        self.scenarios = scenarios

        self.build_map(cfg.map_info)
        return


    def set_subset(self, fraction=None, idxs=None, selected_scenarios_types=None, num_scenarios_per_type=-1):
        dataset_size = len(self)

        sampled_idxs = []
        if fraction != None:
            num_samples = int(dataset_size *fraction)
            num_keep = min(dataset_size, num_samples)
            # sampled_idxs = random.sample(range(dataset_size), num_keep)
            sampled_idxs = np.around(np.linspace(0, dataset_size-1, num_keep)).astype(int)
        elif type(idxs) != type(None):
            sampled_idxs = idxs
        elif selected_scenarios_types != None:
            type_idxs = collections.defaultdict(list)
            for scenario in self.scenarios:
                if scenario.scenario_type in selected_scenarios_types:
                    type_idxs[scenario.scenario_type].append(scenario.data_idx)
            for idxs in type_idxs.values():
                sampled_idxs.extend(idxs[:num_scenarios_per_type])
        else:
            pass

        if len(sampled_idxs) > 0:
            self.subset = True
            self.scenarios = [self.scenarios[idx] for idx in sampled_idxs]
            self._file_names = [self._file_names[idx] for idx in sampled_idxs]
        return


    def __len__(self):
        return len(self.scenarios)
    

    def __getitem__(self, index):
        scenario = self.scenarios[index]
        with open(self._file_names[index], 'rb') as f:
            data = rldev.Data(**pickle.load(f)).option(recursive=True)
        
        # data.pop('timestamp')
        # data.pop('front_camera_image')
        # data.pop('back_camera_image')
        # data.pop('laser1_scan')
        # data.pop('laser2_scan')
        # data.pop('laser3_scan')

        ### find desired row
        map_info = self.cfg.map_info
        max_row = int((map_info.num_rows - 1)/2)
        delta_y = np.linspace(-max_row, max_row, map_info.num_rows) * map_info.row_interval

        y = data.pose[1]
        if y < delta_y[0]:
            start_idx, end_idx = 0, 1
        elif y > delta_y[-1]:
            start_idx, end_idx = -1, -2
        else:
            start_idx, end_idx = np.where(delta_y <= y)[0][-1], np.where(delta_y > y)[0][0]
        data.update(map_polylines=self.map_lines[[start_idx,end_idx]])

        ### find goal
        yaw = data.pose[2]
        if yaw >= -np.pi/2 and yaw < np.pi/2:
            goal_yaw = 0
            front_goal = np.append(data.map_polylines[:, -1, :2].mean(axis=0), goal_yaw).astype(np.float32)
            back_goal = np.append(data.map_polylines[:, 0, :2].mean(axis=0), goal_yaw).astype(np.float32)
        else:
            goal_yaw = np.pi
            front_goal = np.append(data.map_polylines[:, 0, :2].mean(axis=0), goal_yaw).astype(np.float32)
            back_goal = np.append(data.map_polylines[:, -1, :2].mean(axis=0), goal_yaw).astype(np.float32)

        data.update(front_goal=front_goal, back_goal=back_goal)
        data.pose = data.pose.astype(np.float32)
        data.linear_velocity = data.linear_velocity.astype(np.float32)
        data.angular_velocity = data.angular_velocity.astype(np.float32)

        # if self.extra_prediction_dir != None:
        #     extra_pred_path = os.path.join(self.extra_prediction_dir, self.data_type, f'{self.scenarios[index].get_name()}.npz')
        #     extra_predicted_object_trajectories = rldev.Data(**dict(np.load(extra_pred_path)))
        #     data.update(extra_predicted_object_trajectories=extra_predicted_object_trajectories)

        # data = process_data(data, self.max_agents, self.past_ts, self.future_ts)
        return data, scenario


    def collate(self, *args, **kwargs):
        return collate(*args, **kwargs)


    def build_map(self, map_info):
        pillar_length = map_info.cell_length * map_info.num_cells * (map_info.num_pillar - 1) /2


        num_points = int(pillar_length / map_info.resolution)
        base_x = np.concatenate([np.linspace(-pillar_length, -map_info.resolution, num_points), np.linspace(0, pillar_length, num_points+1)])
        base_y = np.zeros_like(base_x)
        base_yaw = np.zeros_like(base_x)
        base_line = np.stack([base_x, base_y, base_yaw], axis=-1)

        max_row = int((map_info.num_rows - 1)/2)
        delta_y = np.linspace(-max_row, max_row, map_info.num_rows) * map_info.row_interval
        delta_x = np.zeros_like(delta_y)
        delta_yaw = np.zeros_like(delta_y)
        delta_line = np.stack([delta_x, delta_y, delta_yaw], axis=-1)

        self.map_lines = (base_line[None] + delta_line[:, None]).astype(np.float32)
        return




def collate(batch):
    data_list = [batch_i[0] for batch_i in batch]
    scenarios = [batch_i[1] for batch_i in batch]

    data = rldev.stack_data(data_list)

    data = data.stack(axis=0)

    return data, scenarios
