import rldev
# rldev.globv.init()

import warnings
import collections
import logging
from typing import Any, Dict
import os
import copy
import pathlib
import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed
from torchmetrics import Metric


from apep.model.template import build_module_wrapper
from apep.dataset.builder import build_datamodule, transfer_batch_to_device
from .tools import model_summary


logger = logging.getLogger(__name__)





class TrainerTemplate(object):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        ################################################################################################
        ### distributed
        self.launcher = cfg.launcher
        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.launcher != None
        if self.distributed:
            self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            torch.cuda.set_device(self.local_rank % self.num_gpus)
            torch.distributed.init_process_group(backend='nccl')
            self.rank = torch.distributed.get_rank()
            self.work_size = torch.distributed.get_world_size()
        else:
            self.local_rank = 0
            self.rank = 0
            self.work_size = 1
        print(f'rank: {self.rank}, local_rank: {self.local_rank}, work_size: {self.work_size}')
        
        rldev.setup_seed(cfg.seed)
        
        ################################################################################################
        ### config
        self.num_epochs = cfg.max_epochs
        if self.local_rank == 0:
            self.ckpt_dir = pathlib.Path(cfg.output_dir) / 'checkpoints'
            self.ckpt_dir.mkdir(exist_ok=True)
        self.val_data_types = list(cfg.val_data_types)
        self.save_ckpt_ratio = cfg.save_ckpt_ratio   ## [0, 1]

        ################################################################################################
        ### counter
        self.epoch_idx = 0
        self.global_step = 0
        self.training = False

        ################################################################################################
        ### metrics
        if self.local_rank == 0:
            self.writer = SummaryWriter(cfg.output_dir)
        self.log_interval = 50
        self.metric_classes = {}

        ################################################################################################
        ### model
        self.device = 'cuda'
        self.init_models()
        self.unwrapped_model = self.model
        self.unwrapped_models = copy.copy(self.models)
        if self.local_rank == 0:
            for key, value in self.unwrapped_models.items():
                model_summary(value)

        self.init_optimizers()


        ################################################################################################
        ### callbacks
        self.init_callbacks()

        ################################################################################################
        ### data
        # batch_size = cfg.data_loader.params.batch_size
        self.datamodule = build_datamodule(cfg)
        self.datamodule.setup('train')
        self.training_dataset = self.datamodule._train_set
        self.validation_datasets = {data_type: self.datamodule._val_sets[data_type] for data_type in self.val_data_types}
        # self.validation_datasets['val14'].set_subset(idxs=[298])

        self.init_distributed_run()
        self.init_params()
        return
    

    def init_models(self):
        self.model = build_module_wrapper(self.cfg.model, self.cfg)
        self.model.to(self.device)

        ### pretrain model
        if self.cfg.pretrain_model_checkpoint_path != None:
            params_pl = torch.load(os.path.expanduser(self.cfg.pretrain_model_checkpoint_path), map_location='cpu')['state_dict']
            params = collections.OrderedDict()
            for key, value in params_pl.items():
                new_key = key[6:]  ### remove 'model.' prefix
                params[new_key] = value
            self.model.load_state_dict(params, strict=False)
            logger.info(f'\n\n\n            load pretrain checkpoint: {os.path.expanduser(self.cfg.pretrain_model_checkpoint_path)}\n\n\n')
        
        ### ckpt
        # if self.cfg.py_func == 'test' or self.cfg.py_func == 'val' or self.cfg.py_func == 'val-vis':
        #     ckpt_path = os.path.expanduser(self.cfg.model_checkpoint_path)
        #     self.load_ckpt(ckpt_path)

        self.models = dict(model=self.model)
        return

    def init_optimizers(self):
        cfg_optimizer = self.cfg.optimizer
        self.optimizer: Optimizer = instantiate(
            config=cfg_optimizer,
            params=self.model.parameters(),
            lr=cfg_optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.3, patience=0, min_lr=3e-5, verbose=True, mode='min')
        self.lr_scheduler = dict(scheduler=scheduler, interval='epoch', monitor=f'loss/{self.val_data_types[0]}_loss')
        return

    def init_callbacks(self):
        cfg_callbacks = self.cfg.callbacks
        # vis_callback = instantiate(cfg_callbacks.vis_callback)
        self.callbacks = [instantiate(cfg_callback) for cfg_callback in cfg_callbacks.values()]
        # self.callbacks = []
        return



    def init_distributed_run(self):
        params_train = copy.deepcopy(self.datamodule._dataloader_params)
        params_val = copy.deepcopy(self.datamodule._dataloader_params)
        if params_val.batch_size_val > 0:
            params_val.batch_size = params_val.batch_size_val
        params_train = dict(params_train)
        params_val = dict(params_val)
        params_train.pop('batch_size_val')
        params_val.pop('batch_size_val')

        training_sampler = torch.utils.data.distributed.DistributedSampler(self.training_dataset) if self.distributed else None
        self.training_sampler = training_sampler

        self.training_dataloader = torch.utils.data.DataLoader(
            dataset=self.training_dataset,
            shuffle=training_sampler == None,
            sampler=training_sampler,
            collate_fn=self.training_dataset.collate,
            **params_train,
        )

        self.validation_dataloaders = {}
        for data_type in self.val_data_types:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(self.validation_datasets[data_type], shuffle=False) if self.distributed else None
            self.validation_dataloaders[data_type] = torch.utils.data.DataLoader(
                dataset=self.validation_datasets[data_type],
                shuffle=False,
                sampler=validation_sampler,
                collate_fn=self.validation_datasets[data_type].collate,
                **params_val,
            )

        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank % self.num_gpus], find_unused_parameters=self.cfg.find_unused_parameters)
            self.models.update(model=self.model)
        return

    def init_params(self):
        pass


    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if self.unwrapped_model.__class__.__name__ == 'MTR':
            # self.model.load_params_from_file(self.cfg.model_checkpoint_path, logger=logger, to_cpu=True)
            raise NotImplementedError
            self.unwrapped_model.load_state_dict(checkpoint['model_state'], strict=True)
        else:
            for key, model in self.unwrapped_models.items():
                if key in checkpoint:
                    model.load_state_dict(checkpoint[key], strict=True)
                    logger.info(f'        load checkpoint {key}: {ckpt_path}')
                else:
                    # Warn instead of throwing an error if the key is not found
                    warning_message = f"Key '{key}' not found in checkpoint. Model '{key}' state dict not loaded."
                    warnings.warn(warning_message)
                    logger.warning(warning_message)
            if 'model' in checkpoint:
                self.unwrapped_model.load_state_dict(checkpoint['model'], strict=True)
            else:
                raise NotImplementedError
        logger.info(f'\n\n\n        load checkpoint: {ckpt_path}\n\n\n')
        return



    def train(self):
        for epoch_idx in range(self.num_epochs):
            self.epoch_idx = epoch_idx
            self.epoch_start()

            ### train
            self.training_start()
            if self.local_rank == 0:
                tepoch = tqdm.tqdm(self.training_dataloader, unit="batch", desc=f"Epoch {epoch_idx} [TRAIN]", leave=False)
            else:
                tepoch = self.training_dataloader
            for i, batch in enumerate(tepoch):
                batch = self.transfer_batch_to_device(batch)
                loss = self.training_step(batch, i)
                # logger.info(f'rank {self.local_rank}, epoch {self.epoch_idx}, batch {i}: {[scenario.data_idx for scenario in batch[1]]}')
                # logger.info(f'rank {self.local_rank}, epoch {self.epoch_idx}, batch {i}: loss {loss.item()}')
                if i > 0 and i % int(len(tepoch) * self.save_ckpt_ratio) == 0:
                    self.save_model(epoch_idx -1 + i / len(tepoch))
                self.global_step += 1
                if self.local_rank == 0:
                    tepoch.set_postfix(loss="{:.3f}".format(loss.item()))
            if self.local_rank == 0:
                tepoch.close()
            self.training_end()

            ### eval
            self.validation_start()
            for data_type in self.val_data_types:
                if self.local_rank == 0:
                    tepoch = tqdm.tqdm(self.validation_dataloaders[data_type], unit="batch", desc=f"Epoch {epoch_idx} [VAL: {data_type}]", leave=False)
                else:
                    tepoch = self.validation_dataloaders[data_type]
                for i, batch in enumerate(tepoch):
                    batch = self.transfer_batch_to_device(batch)
                    loss = self.validation_step(batch, i, prefix=data_type)
                if self.local_rank == 0:
                    tepoch.close()
            self.validation_end()

            ### save model
            self.save_model(epoch_idx)
        return


    @torch.no_grad()
    def validate(self, all=True):
        ckpt_path = os.path.join(os.path.expanduser(self.cfg.ckpt_dir), f'epoch={self.cfg.ckpt_num}.ckpt')
        start_epoch = torch.load(ckpt_path)['epoch_idx']
        ckpt_dir, ckpt_name = os.path.split(ckpt_path)
        if all:
            s1, s2 = ckpt_name.split(str(start_epoch))
            end_epoch = max([int(name.split(s1)[-1].split(s2)[0]) for name in os.listdir(ckpt_dir)])
        else:
            end_epoch = start_epoch

        for epoch_idx in range(start_epoch, end_epoch+1):
            self.epoch_idx = epoch_idx
            self.epoch_start()

            self.load_ckpt(os.path.join(ckpt_dir, ckpt_name.replace(str(start_epoch), str(epoch_idx))))
            
            ### eval
            self.validation_start()
            for data_type in self.val_data_types:
                if self.local_rank == 0:
                    tepoch = tqdm.tqdm(self.validation_dataloaders[data_type], unit="batch", desc=f"[VAL: {data_type}]", leave=False)
                else:
                    tepoch = self.validation_dataloaders[data_type]
                for i, batch in enumerate(tepoch):
                    batch = self.transfer_batch_to_device(batch)
                    loss = self.validation_step(batch, i, prefix=data_type)
                if self.local_rank == 0:
                    tepoch.close()
            self.validation_end()
            if self.local_rank == 0:
                logger.info(f'Metrics:')
                for tag, metric_class in self.metric_classes.items():
                    metric_value = metric_class.computor.compute()  # Make sure this is the final metric value after validation
                    logger.info(f'    {tag}: {metric_value.item():.6f}')
        return



    def epoch_start(self):
        if self.training_sampler != None:
            self.training_sampler.set_epoch(self.epoch_idx)
        return

    def training_start(self):
        torch.cuda.empty_cache()
        [model.train() for model in self.models.values()]
        [objective.train() for model in self.unwrapped_models.values() for objective in model.objectives.values()]
        [metric.train() for model in self.unwrapped_models.values() for metric in model.metrics.values()]
        [metric.train() for model in self.unwrapped_models.values() for metric in model.quality_metrics.values()]
        self.training = True
        return

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, 'train')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def training_end(self):
        return



    @torch.no_grad()
    def validation_start(self):
        torch.cuda.empty_cache()
        [model.eval() for model in self.models.values()]
        [objective.eval() for model in self.unwrapped_models.values() for objective in model.objectives.values()]
        [metric.eval() for model in self.unwrapped_models.values() for metric in model.metrics.values()]
        [metric.eval() for model in self.unwrapped_models.values() for metric in model.quality_metrics.values()]
        self.training = False
        
        [metric_class.computor.reset() for metric_class in self.metric_classes.values()]
        [metric_class.saver.reset() for metric_class in self.metric_classes.values()]

        idxs = None
        [callback.on_validation_start(self, idxs=idxs) for callback in self.callbacks if hasattr(callback, 'on_validation_start')]
        return


    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, prefix) -> torch.Tensor:
        return self.step(batch, prefix)

    @torch.no_grad()
    def validation_end(self):
        val_loss = None
        for tag, metric_class in self.metric_classes.items():
            metric = metric_class.computor.compute()
            if tag == self.lr_scheduler['monitor']:
                val_loss = metric
            if self.local_rank == 0:
                self.writer.add_scalar(tag, metric, self.epoch_idx)
        
        self.lr_scheduler['scheduler'].step(val_loss)
        if self.local_rank == 0:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(f'lr-{self.optimizer.__class__.__name__}', lr, self.epoch_idx)
        
        # idxs = None
        # idxs, data = self.metric_classes['traj_benchmark_metrics_val/drivable_area_compliance'].saver.get()
        # idxs = idxs[torch.where(data < 1)]
        # [callback.on_validation_end(self, list(self.validation_datasets.values())[0], prefix=self.val_data_types[0], idxs=idxs) for callback in self.callbacks if hasattr(callback, 'on_validation_end')]
        [callback.on_validation_end(self) for callback in self.callbacks if hasattr(callback, 'on_validation_end')]
        return


    def save_model(self, idx):
        idx = np.round(idx, 4)
        if self.local_rank == 0:
            model_states = {key: model.state_dict() for key, model in self.unwrapped_models.items()}
            torch.save({**model_states, **dict(epoch_idx=idx)}, self.ckpt_dir / f'epoch={idx}.ckpt')
        return



    def step(self, batch, prefix: str) -> torch.Tensor:
        output_data, objectives, metrics, timewise_metrics, benchmark_metrics = self.forward(batch, prefix)
        loss = aggregate_objectives(objectives)

        data, scenarios = batch
        idxs = torch.tensor([scenario.data_idx for scenario in scenarios])

        self.log_all(idxs, loss, objectives, metrics, timewise_metrics, benchmark_metrics, prefix)
        if not self.training:
            [callback.on_validation_step(self, batch, output_data, prefix) for callback in self.callbacks if hasattr(callback, 'on_validation_step')]
        return loss.mean()


    def forward(self, batch, prefix):
        raise NotImplementedError


    def forward_model(self, data: rldev.Data, scenarios):
        output_data = self.model(data, scenarios)
        return output_data




    def compute_metrics(self, input_data, output_data, targets) -> Dict[str, torch.Tensor]:
        metrics_res = {}
        for metric_name, metric in self.unwrapped_model.metrics.items():
            if not metric.training == self.training:
                continue
            res = metric.compute(input_data, output_data, targets)
            if isinstance(res, dict):
                metrics_res.update(res)
            else:
                metrics_res[metric_name] = res
        return metrics_res

    def compute_timewise_metrics(self, input_data, output_data, targets) -> Dict[str, torch.Tensor]:
        return rldev.Data(**{metric_name: metric.compute(input_data, output_data, targets) for metric_name, metric in self.unwrapped_model.timewise_metrics.items() if metric.training == self.training})

    def compute_quality_metrics(self, input_data, output_data, targets) -> Dict[str, torch.Tensor]:
        return rldev.Data(**{metric_name: metric.compute(input_data, output_data, targets) for metric_name, metric in self.unwrapped_model.quality_metrics.items()})





    def transfer_batch_to_device(self, batch):
        return transfer_batch_to_device(batch, self.device)

    
    def log_all(
        self,
        idxs,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        timewise_metrics: rldev.Data,
        benchmark_metrics: rldev.Data,
        prefix: str,
        loss_name: str = 'loss',
    ) -> None:

        self.log(idxs, f'loss/{prefix}_{loss_name}', loss)

        for key, value in objectives.items():
            self.log(idxs, f'objectives_{prefix}/{key}', value)

        for key, value in metrics.items():
            self.log(idxs, f'metrics_{prefix}/{key}', value)

        # for key, value in timewise_metrics.mean(dim=ObjectTensorOutputSize.mode).to_dict().items():
        #     num_timesteps = value.shape[ObjectTensorOutputSize.time-2]
        #     for t in range(num_timesteps):
        #         self.log(idxs, f'timewise_metrics_{prefix}/{key}_time_{t}', value[:, t])

        for key, value in benchmark_metrics.to_dict().items():
            self.log(idxs, f'traj_benchmark_metrics_{prefix}/{key}', value)
        return
        

    def log(self, idxs, tag, value):
        if self.training:
            if self.global_step % self.log_interval != 0:
                return
            if self.local_rank != 0:
                return
        
            self.writer.add_scalar(tag, value.detach().cpu().mean().item(), self.global_step)
        
        else:
            if not tag in self.metric_classes:
                metric_class = rldev.Data(computor=GeneralMetric(name=tag).to(self.device), saver=MetricSaver(tag))
                self.metric_classes[tag] = metric_class
            else:
                metric_class = self.metric_classes[tag]
            
            metric_class.computor(value.detach())
            metric_class.saver.add(idxs, value.detach())
        return








def aggregate_objectives(objectives):
    return torch.stack(list(objectives.values()), dim=-1).sum(dim=-1)





class GeneralMetric(Metric):
    full_state_update = False
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss):
        """
            loss: torch.Size([batch_size])
        """
        self.loss += loss.sum()
        self.total += loss.shape[0]
        return

    def compute(self):
        return self.loss.float() / self.total
    

class MetricSaver(object):
    def __init__(self, name) -> None:
        self.name = name
        self.idxs = []
        self.data = []
    
    def add(self, idxs, data):
        self.idxs.append(idxs.cpu())
        self.data.append(data.detach().cpu())

    def get(self):
        return torch.cat(self.idxs, dim=0), torch.cat(self.data, dim=0)

    def reset(self):
        self.idxs = []
        self.data = []
        return

