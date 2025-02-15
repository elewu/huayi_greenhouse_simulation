import rldev

from typing import Dict, List, cast

import torch




class AbstractMetric:
    def __init__(self, name, fix_mode=False) -> None:
        self.name = name
        self.fix_mode = fix_mode
        self.training = False


    def train(self):
        if not self.fix_mode:
            self.training = True
    def eval(self):
        if not self.fix_mode:
            self.training = False


