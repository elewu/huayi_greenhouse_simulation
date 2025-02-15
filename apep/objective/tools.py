import rldev

from typing import Dict, List, cast

import torch
import torch.nn as nn
import torch.nn.functional as F




class AbstractObjective:
    def __init__(self, name, weight, scenario_type_loss_weighting: Dict[str, float]={'unknown': 1.0}) -> None:
        super().__init__()
        self.name = name
        self.weight = weight
        self.training = False

        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def train(self):
        self.training = True
    def eval(self):
        self.training = False
