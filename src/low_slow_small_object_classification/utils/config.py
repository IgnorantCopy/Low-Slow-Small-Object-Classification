import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import lightning as pl

from src.low_slow_small_object_classification.utils.logger import default_logger as logger


def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
    return config


def save_config(config, config_file):
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def override_config(src_config, dst_config):
    for key, value in src_config.items():
        dst_config[key] = value

def get_model(config) -> pl.LightningModule:
    model_name = config['model']['name']
    logger.info(f'Building model {model_name}')

    if model_name == 'Swin3D':
        from src.low_slow_small_object_classification.models.swin3d import Swin3D
        model = Swin3D(config)
    elif model_name == 'MultiRocket':
        from src.low_slow_small_object_classification.models.multi_rocket import MultiRocket
        model = MultiRocket(config)
    elif model_name == 'Stacking':
        from src.low_slow_small_object_classification.models.stacking import StackingModule
        model = StackingModule(config)
    else:
        raise ValueError(f'Model {model_name} not supported')

    return model

def get_optimizer(train_config, model: nn.Module) -> optim.Optimizer:
    optimizer_config = train_config["optimizer"]
    optimizer_name = optimizer_config["name"]
    lr = train_config["lr"]

    logger.info(f"Loading optimizer: {optimizer_name}")
    if optimizer_name == "Adam":
        weight_decay = optimizer_config.get("weight_decay", 5e-4)
        betas = optimizer_config.get("betas", (0.9, 0.999))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_name == "SGD":
        momentum = optimizer_config.get("momentum", 0.9)
        weight_decay = optimizer_config.get("weight_decay", 5e-4)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        logger.error(f"Unsupported optimizer: {optimizer_name}")
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    logger.info(f"{optimizer_name} optimizer loaded")
    return optimizer

def get_scheduler(train_config, optimizer: optim.Optimizer):
    scheduler_config = train_config["scheduler"]
    scheduler_name = scheduler_config["name"]

    logger.info(f"Loading scheduler: {scheduler_name}")
    if scheduler_name == "ReduceLROnPlateau":
        factor = scheduler_config.get("factor", 0.5)
        patience = scheduler_config.get("patience", 10)
        min_lr = scheduler_config.get("min_lr", 1e-6)
        scheduler =  lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
    elif scheduler_name == "MultiStepLR":
        milestones = scheduler_config["milestones"]
        gamma = scheduler_config.get("gamma", 0.5)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        T_0 = scheduler_config.get("T_0", 100)
        eta_min = scheduler_config.get("eta_min", 1e-6)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)
    else:
        logger.error(f"Unsupported scheduler: {scheduler_name}")
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logger.info(f"{scheduler_name} scheduler loaded")
    return scheduler


def get_datamodule(data_config):
    data_name = data_config['name']
    logger.info(f'Building datamodule {data_name}')
    if data_name == 'rd':
        from src.low_slow_small_object_classification.data.datasets import RDDataModule
        datamodule = RDDataModule(data_config)
    elif data_name == "track":
        from src.low_slow_small_object_classification.data.datasets import TrackDataModule
        datamodule = TrackDataModule(data_config)
    elif data_name == "multi":
        from src.low_slow_small_object_classification.data.datasets import MultiDataModule
        datamodule = MultiDataModule(data_config)
    else:
        raise ValueError(f'Datamodule {data_name} not supported')

    return datamodule