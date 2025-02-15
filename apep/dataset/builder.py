import rldev

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import copy
import numpy as np
from omegaconf import DictConfig

import torch



def build_datamodule(cfg: DictConfig):
    # Create data augmentation
    # augmentors = build_agent_augmentor(cfg.data_augmentation) if 'data_augmentation' in cfg else None

    # Create datamodule
    datamodule = DataModule(
        cfg=cfg,
        dataloader_params=cfg.data_loader.params,
        # augmentors=augmentors,
        # scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
        **cfg.data_loader.datamodule,
    )

    return datamodule



class DataModule:
    def __init__(self, cfg: DictConfig, train_fraction: float, val_fraction: float, test_fraction: float, dataloader_params: Dict[str, Any]):
        self.cfg = cfg
        # self.load_train_set = cfg.load_train_set  ### only in validate
        self.dataset_name = cfg.dataset_name
        self.val_data_types = cfg.val_data_types
        self.data_dir = os.path.expanduser(cfg.data_dir)
        self.train_split = cfg.train_split

        if self.dataset_name == 'greenhouse':
            from .greenhouse import create_dataset
            self.create_dataset = create_dataset
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        # Fractions
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        # Data loader for train/val/test
        self._dataloader_params = dataloader_params

        # self.metric_engines, self.batch_metric_engine = build_metrics_engines(cfg, all_scenarios)
        return


    def setup(self, stage: Optional[str] = None) -> None:

        if stage is None:
            return

        if stage == 'train':
            self._train_set = self.create_dataset(self.cfg, self.data_dir, self._train_fraction, self.train_split)
            self._val_sets = {data_type: self.create_dataset(self.cfg, self.data_dir, self._val_fraction, data_type) for data_type in self.val_data_types}

            scenarios = []
            scenarios.extend(self._train_set.scenarios)
            for val_set in self._val_sets.values():
                scenarios.extend(val_set.scenarios)

        elif stage == 'val':
            self._val_sets = {data_type: self.create_dataset(self.cfg, self.data_dir, self._val_fraction, data_type) for data_type in self.val_data_types}
            scenarios = []
            for val_set in self._val_sets.values():
                scenarios.extend(val_set.scenarios)

        else:
            raise ValueError(f'Stage must be one of ["train", "val"], got ${stage}.')
        
        # print_scenario_types(scenarios)
        # self.metric_engines, self.batch_metric_engine = build_metrics_engines(self.cfg, scenarios)
        self.metric_engines, self.batch_metric_engine = None, None
        return




def transfer_batch_to_device(batch, device: torch.device):
    data: rldev.Data = batch[0]
    scenarios = batch[1]

    data = copy.copy(data)

    data = data.to_tensor().to(device)

    return data, scenarios

