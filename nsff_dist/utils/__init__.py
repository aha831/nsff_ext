# optimizer
from torch.optim import SGD, Adam
import torch
import torch_optimizer as optim
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *

def get_named_parameters(models):
    """Get all model parameters recursively."""
    named_parameters = []
    if isinstance(models, list):
        for model in models:
            named_parameters += get_named_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            named_parameters += get_named_parameters(model)
    else: # models is actually a single pytorch model
        named_parameters += list(models.named_parameters())
    return named_parameters
    
def get_optimizer(hparams, models, **kwargs):
    """
    kwargs: {prefix: lr}
    """
    eps = 1e-8
    named_parameters = get_named_parameters(models)

    if kwargs:
        parameters = []
        for n, p in named_parameters:
            for prefix, lr in kwargs.items():
                if n.startswith(prefix):
                    parameters += [{'params': p, 'lr': lr}]
                else:
                    parameters += [{'params': p, 'lr': hparams.lr}]
    else:
        parameters = [p for n, p in named_parameters]

    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr, 
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps, 
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = optim.RAdam(parameters, lr=hparams.lr, eps=eps, 
                                weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = optim.Ranger(parameters, lr=hparams.lr, eps=eps, 
                                 weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer, 
                             lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, 
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path:
        return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict, strict=False)