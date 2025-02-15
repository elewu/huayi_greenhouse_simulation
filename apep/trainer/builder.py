

import logging
from hydra._internal.utils import _locate
from omegaconf import DictConfig, OmegaConf

import torch

from apep.trainer.template import TrainerTemplate


logger = logging.getLogger(__name__)




def scale_parameter(
    parameter: float, world_size: int, equal_variance_scaling_strategy: bool, raise_power: bool = False
) -> float:
    """
    Scale parameter (such as learning rate or beta values in Adam/AdamW optimizer) using method specified in the context of PytorchLightning's ddp.
    :param parameter: Learning rate/beta values used in Adam optimizer/etc.
    :param world_size: Number gpus used.
    :param equal_variance_scaling_strategy: Whether the method to scale the learning rate or betas by is equal_variance (by square root of num GPUs); otherwise it is linearly (by num GPUs).
    :return parameter: Learning rate/beta values used in Adam optimizer/etc after scaling.
    """
    scaling_factor = world_size**0.5 if equal_variance_scaling_strategy else world_size
    parameter = parameter * scaling_factor if not raise_power else parameter**scaling_factor

    return parameter



def update_distributed_optimizer_config(cfg: DictConfig) -> DictConfig:
    """
    Scale the learning rate according to scaling method provided in distributed setting with ddp strategy.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return cfg: DictConfig. Updated configuration that is used to run the experiment.
    """
    lr_scale = torch.cuda.device_count()
    logger.info(f"World size: {lr_scale}")
    logger.info(f"Learning rate before: {cfg.optimizer.lr}")

    equal_variance_scaling_strategy = True
    scaling_method = (
        "Equal Variance" if equal_variance_scaling_strategy else "Linearly"
    )
    logger.info(f"Scaling method: {scaling_method}")

    # TODO: support other distributed training strategies like ddp2, dp, etc
    cfg.optimizer.lr = scale_parameter(
        parameter=cfg.optimizer.lr,
        world_size=lr_scale,
        equal_variance_scaling_strategy=equal_variance_scaling_strategy,
    )

    cfg.optimizer.betas[0] = scale_parameter(
        parameter=cfg.optimizer.betas[0],
        world_size=lr_scale,
        equal_variance_scaling_strategy=equal_variance_scaling_strategy,
        raise_power=True,
    )
    cfg.optimizer.betas[1] = scale_parameter(
        parameter=cfg.optimizer.betas[1],
        world_size=lr_scale,
        equal_variance_scaling_strategy=equal_variance_scaling_strategy,
        raise_power=True,
    )
    logger.info(f"Betas after scaling: {cfg.optimizer.betas}")

    logger.info(f"Learning rate after scaling: {cfg.optimizer.lr}")

    return cfg




def build_trainer(cfg: DictConfig) -> TrainerTemplate:
    logger.info('Building trainer...')

    # ### ! todo
    OmegaConf.set_struct(cfg, False)
    cfg = update_distributed_optimizer_config(cfg)
    OmegaConf.set_struct(cfg, True)

    # # Build trainer
    # if cfg.job_name in ['invalid']:
    #     raise ValueError
    
    # elif cfg.job_name in ['apep']:
    #     module_cls = TrainerPPO

    # else:
    #     module_cls = TrainerIL
    
    if hasattr(cfg.model.model_params, 'trainer_cls'):
        module_cls = _locate(cfg.model.model_params.trainer_cls)
    else:
        raise NotImplementedError
    trainer = module_cls(cfg)
    return trainer

