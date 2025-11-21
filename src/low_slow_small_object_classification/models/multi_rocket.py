import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import numpy as np
from tsai.models.MultiRocketPlus import MultiRocketBackbonePlus
from typing import Dict, Tuple

from src.low_slow_small_object_classification.data.track_preprocessor import TrajectoryPreprocessor
from src.low_slow_small_object_classification.utils.logger import default_logger as logger
from src.low_slow_small_object_classification.utils.config import get_optimizer, get_scheduler
from src.low_slow_small_object_classification.models.utils import calc_rate


np.seterr(all='raise')

# region MultiRocket

class MultiRocketModel(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, num_features: int = 20_000, dropout: float = 0.2, **kwargs):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.num_features = num_features
        self.dropout = dropout

        self.backbone = MultiRocketBackbonePlus(c_in, seq_len, num_features, **kwargs)
        backbone_out_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(backbone_out_features),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_features, c_out),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out)
        )

    def forward(self, x: torch.Tensor, last_logits: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        fea = self.backbone(x)
        x = self.fc(fea)
        if last_logits is None:
            last_logits = torch.ones((batch_size, self.c_out), dtype=x.dtype, device=x.device) / self.c_out
        x = x + last_logits
        x = self.head(x)
        return x, fea


class MultiRocketClassifier(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, num_features: int = 20_000,
                 dropout: float = 0.2, confidence_threshold: float = 0.9, **kwargs):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.num_features = num_features
        self.dropout = dropout
        self.confidence_threshold = confidence_threshold
        self.support_lengths = self._get_support_lengths()
        self.models = nn.ModuleList([])

        for i in range(1, self.seq_len + 1):
            length = self._find_support_length(i)
            self.models.append(MultiRocketModel(
                c_in=c_in,
                c_out=c_out,
                seq_len=length,
                num_features=num_features,
                dropout=dropout,
                **kwargs,
            ))

    def _get_support_lengths(self):
        lengths = [i for i in range(10, self.seq_len + 1)]
        lengths.remove(17)
        lengths.remove(25)
        return lengths

    def _find_support_length(self, length):
        for i in self.support_lengths:
            if i >= length:
                return i
        return self.support_lengths[-1]

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, seq_len = x.shape
        if seq_len < 10:
            temp = torch.zeros((batch_size, num_features, 10)).to(x)
            temp[:, :, :seq_len] = x
            temp[:, -2, seq_len:] = 1
            temp[:, -1, :] = -1
            x = temp

        target_len = self._find_support_length(seq_len)
        if x.shape[2] < target_len:
            padded_data = []
            for i in range(batch_size):
                pad_data = x[i].cpu().numpy().T
                pad_data = TrajectoryPreprocessor.data_padding(pad_data, target_len, n=1).T
                padded_data.append(torch.from_numpy(pad_data).to(x))
            x = torch.cat([x, torch.stack(padded_data)], dim=2)

        return x

    def forward(self, x: torch.Tensor, last_logits: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        seq_len = x.shape[2]
        x = self.preprocess(x)
        logits, features = self.models[seq_len - 1](x, last_logits)

        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "max_probs": max_probs,
            "features": features,
        }

# endregion

# region Lightning Module

class MultiRocket(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config["model"]
        model_config.pop("name", None)
        self.model = MultiRocketClassifier(**model_config)
        self.automatic_optimization = False

        self.seq_len = model_config["seq_len"]
        self.criterion = nn.CrossEntropyLoss()
        self.accuracies = torch.zeros(self.seq_len)
        self.count = 0
        self.num_classes = model_config["c_out"]
        self.conf_matrix = torch.zeros((self.num_classes, self.num_classes))

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizers = []
        schedulers = []
        for i in range(self.seq_len):
            optimizer = get_optimizer(train_config, self.model.models[i])
            optimizers.append(optimizer)
            schedulers.append(get_scheduler(train_config, optimizer))

        return optimizers, schedulers

    def update_conf_matrix(self, predictions, labels):
        for i in range(len(predictions)):
            pred = predictions[i]
            unique_values, counts = torch.unique(pred, return_counts=True)
            pred_label = unique_values[counts.argmax()]
            label = labels[i]
            self.conf_matrix[int(label), int(pred_label)] += 1

    def forward(self, t, sequence, last_logits):
        x = sequence[:, :, :t]
        output = self.model(x, last_logits)
        return output

    def step(self, batch, require_conf=False):
        sequence = batch["sequence"].transpose(1, 2)
        labels = batch["label"]
        last_logits = torch.ones((sequence.shape[0], self.num_classes)).to(sequence) / self.num_classes

        b = sequence.shape[0]
        accuracy = torch.zeros(self.seq_len)
        accuracy_by_class = torch.zeros(self.num_classes, self.seq_len)
        begin_time = torch.ones(b) * self.seq_len
        begin = [False for _ in range(b)]
        prediction = torch.zeros((b, self.seq_len))
        total_loss = 0.
        loss_by_step = []
        for t in range(1, self.seq_len + 1):
            output = self.forward(t, sequence, last_logits)
            logits = output["logits"]
            probs = output["probs"]
            last_logits = logits
            loss = self.criterion(probs, labels)
            total_loss += loss
            loss_by_step.append(loss.item())

            _, pred = torch.max(logits, 1)
            prediction[:, t - 1] = pred
            accuracy[t - 1] = (pred == labels).float().mean()
            sequence[:, -1, t - 1] = pred.float()
            for i in range(self.num_classes):
                accuracy_by_class[i, t - 1] = (pred[labels == i] == i).float().mean().item()

            for i in range(b):
                if not begin[i] and pred[i] == labels[i]:
                    begin_time[i] = t
                    begin[i] = True

        rate, strict_prediction = calc_rate(prediction)
        strict_prediction = strict_prediction.to(labels)
        strict_accuracy = (strict_prediction == labels).float().mean().item()

        accuracy = torch.nan_to_num(accuracy, nan=0.0)
        accuracy_by_class = torch.nan_to_num(accuracy_by_class, nan=0.0)

        result = {
            "loss": total_loss,
            "loss_by_step": loss_by_step,
            "begin_time": begin_time.mean(),
            "rate": rate,
            "accuracy": accuracy,
            "accuracy_by_class": accuracy_by_class.mean(1),
            "strict_accuracy": strict_accuracy,
        }

        if require_conf:
            self.update_conf_matrix(prediction, labels)
        result["conf_matrix"] = self.conf_matrix.clone()
        return result

    def on_train_epoch_start(self):
        logger.info(f"Epoch {self.current_epoch} starts.")
        logger.info(f"Start training stage.")
        self.accuracies = torch.zeros(self.seq_len)
        self.count = 0
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)

    def training_step(self, batch, batch_idx):
        b = batch["sequence"].shape[0]
        result = self.step(batch)

        self.loss = result["loss"]
        self.loss_by_step = result["loss_by_step"]
        begin_time = result["begin_time"]
        rate = result["rate"]
        accuracy = result["accuracy"]
        accuracy_by_class = result["accuracy_by_class"]
        strict_accuracy = result["strict_accuracy"]
        self.accuracies += b * accuracy
        self.count += b

        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(self.loss)
        for optimizer in optimizers:
            optimizer.step()

        self.log_dict({
            "train/loss": self.loss,
            "train/avg_accuracy": accuracy.mean(),
            "train/strict_accuracy": strict_accuracy,
            "train/begin_time": begin_time.mean(),
            "train/rate": rate,
        }, on_step=False, on_epoch=True, batch_size=b)
        for i in range(len(accuracy_by_class)):
            self.log(f"train/acc_cls_{i}", accuracy_by_class[i], on_step=False, on_epoch=True, batch_size=b)

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        lrs = []
        for i, scheduler in enumerate(schedulers):
            scheduler.step(self.loss_by_step[i])
            lrs.append(scheduler.get_last_lr()[0])
        self.logger.experiment.add_scalars("train/accuracy",
                                           {f"t{i + 1}": self.accuracies[i] for i in range(0, self.seq_len, 3)},
                                           self.current_epoch)
        self.logger.experiment.add_scalars("lr", {f"t{i + 1}": lrs[i] for i in range(0, self.seq_len, 3)},
                                          self.current_epoch)

    def on_validation_epoch_start(self):
        logger.info(f"Start validation stage.")
        self.accuracies = torch.zeros(self.seq_len)
        self.count = 0
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)

    def validation_step(self, batch, batch_idx):
        b = batch["sequence"].shape[0]
        result = self.step(batch)

        self.loss = result["loss"]
        begin_time = result["begin_time"]
        rate = result["rate"]
        accuracy = result["accuracy"]
        accuracy_by_class = result["accuracy_by_class"]
        strict_accuracy = result["strict_accuracy"]
        self.accuracies += b * accuracy
        self.count += b

        self.log_dict({
            "val/loss": self.loss,
            "val/avg_accuracy": accuracy.mean(),
            "val/strict_accuracy": strict_accuracy,
            "val/begin_time": begin_time.mean(),
            "val/rate": rate,
        }, on_step=False, on_epoch=True, batch_size=b)
        for i in range(len(accuracy_by_class)):
            self.log(f"val/acc_cls_{i}", accuracy_by_class[i], on_step=False, on_epoch=True, batch_size=b)

    def on_validation_epoch_end(self):
        self.accuracies /= self.count
        self.logger.experiment.add_scalars("val/accuracy",
                                           {f"t{i + 1}": self.accuracies[i] for i in range(0, self.seq_len, 3)},
                                           self.current_epoch)

    def on_fit_start(self):
        logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        logger.info(f"All training finished.")

    def on_test_epoch_start(self):
        self.accuracies = torch.zeros(self.seq_len)
        self.count = 0
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)

    def test_step(self, batch, batch_idx):
        b = batch["sequence"].shape[0]
        result = self.step(batch, require_conf=True)

        self.loss = result["loss"]
        begin_time = result["begin_time"]
        rate = result["rate"]
        accuracy = result["accuracy"]
        strict_accuracy = result["strict_accuracy"]
        accuracy_by_class = result["accuracy_by_class"]
        self.accuracies += b * accuracy
        self.count += b

        self.log_dict({
            "test/loss": self.loss,
            "test/avg_accuracy": accuracy.mean(),
            "test/strict_accuracy": strict_accuracy,
            "test/begin_time": begin_time.mean(),
            "test/rate": rate,
        }, on_step=True, on_epoch=True, batch_size=b)
        for i in range(len(accuracy_by_class)):
            self.log(f"test/acc_cls_{i}", accuracy_by_class[i], on_step=True, on_epoch=True, batch_size=b)

    def on_test_epoch_end(self):
        self.accuracies /= self.count
        self.logger.experiment.add_scalars("test/accuracy",
                                           {f"t{i + 1}": self.accuracies[i] for i in range(0, self.seq_len, 3)},
                                           self.current_epoch)

    def on_test_start(self):
        logger.info("Start testing.")

    def on_test_end(self):
        logger.info(f"Testing finished.")
