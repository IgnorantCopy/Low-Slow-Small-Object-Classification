import os
import argparse
import datetime
import lightning as pl
from lightning.pytorch import loggers

from src.low_slow_small_object_classification.utils.config import load_config, get_model, get_datamodule
from src.low_slow_small_object_classification.utils import logger as log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to the checkpoint file")
    parser.add_argument("--device", type=str, default='gpu', help="device to use, default is 'gpu'", choices=['gpu', 'cpu'])

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ckpt = args.ckpt
    device = args.device

    config_path = os.path.join(os.path.dirname(ckpt), "../../../config.yaml")
    config = load_config(config_path)
    model_config, data_config = config["model"], config["data"]
    pl.seed_everything(getattr(data_config, "seed", 42))

    log_dir = os.path.join(os.path.dirname(ckpt), "../../../..", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    logger = loggers.TensorBoardLogger(save_dir=log_dir)
    log.default_logger = log.get_logger(os.path.join(log_dir, "test.log"))

    model = get_model(config)
    model = model.__class__.load_from_checkpoint(ckpt, config=config)
    datamodule = get_datamodule(data_config)
    datamodule.setup("test")

    trainer = pl.Trainer(accelerator=device, devices="auto", logger=logger, default_root_dir=log_dir)
    trainer.test(model=model, dataloaders=datamodule.val_dataloader())


if __name__ == '__main__':
    main()