
import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

    return



def unfreeze_params(model: nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True

    return



def model_summary(model: nn.Module):
    summary = []
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        trainable_module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        summary.append({
            'name': name,
            'type': module.__class__.__name__,
            'params': module_params / 1e3  # Convert to Kilo (K)
        })
        
        total_params += module_params
        trainable_params += trainable_module_params
        non_trainable_params += module_params - trainable_module_params

    # Calculate total size in MB (assuming 4 bytes per parameter)
    total_size_mb = (total_params * 4) / (1024 ** 2)
    
    # Add a total row
    summary.append({
        'name': '',
        'type': '',
        'params': '\n---------------------------------------------'
    })
    summary.append({
        'name': f'{trainable_params / 1e3:.1f} K',
        'type': 'Trainable params',
        'params': ''
    })
    summary.append({
        'name': f'{non_trainable_params / 1e3:.1f} K',
        'type': 'Non-trainable params',
        'params': ''
    })
    summary.append({
        'name': f'{total_params / 1e3:.1f} K',
        'type': 'Total params',
        'params': ''
    })
    summary.append({
        'name': f'{total_size_mb:.3f} MB',
        'type': 'Total estimated model params size',
        'params': ''
    })
    
    # Log the summary
    logger.info('\n')
    logger.info(f"{'Name':<20} | {'Type':<30} | {'Params'}")
    logger.info('-' * 65)
    for item in summary:
        logger.info(f"{item['name']:<20} | {item['type']:<30} | {item['params']}")
    logger.info('\n')
        