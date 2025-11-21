import torch
import torch.nn as nn
import lightning as pl
from thop import profile
from typing import Optional

from src.low_slow_small_object_classification.utils.logger import default_logger as logger
from src.low_slow_small_object_classification.utils.config import get_optimizer, get_scheduler
from src.low_slow_small_object_classification.models.utils import calc_rate
from src.low_slow_small_object_classification.models.multi_rocket import MultiRocketClassifier
from src.low_slow_small_object_classification.models.swin3d import SwinTransformer3D


class StackingModel(nn.Module):
    def __init__(self, swin: SwinTransformer3D, rocket: MultiRocketClassifier,
                 hidden_channels: int, num_classes: int, freeze_stage: int):
        super().__init__()

        self.num_classes = num_classes
        self.freeze_stage = freeze_stage
        self.swin = swin
        self.rocket = rocket
        self.classifier = nn.Sequential(
            nn.Linear(2 * num_classes, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, num_classes),
            nn.Softmax(dim=1),
        )
        self.freeze()

    def freeze(self):
        if self.freeze_stage == 1:
            self.swin.requires_grad_(False)
            self.rocket.requires_grad_(False)
        elif self.freeze_stage == 2:
            self.requires_grad_(False)

    def forward(self, sequences, rd_matrices=None, extra_features=None, mask=None, last_logits=None):
        if rd_matrices is None:
            return self.rocket(sequences, last_logits)
        swin_out = self.swin(rd_matrices, extra_features, mask)[0]
        rocket_out = self.rocket(sequences, last_logits)[0]
        x = self.classifier(torch.cat([rocket_out, swin_out], dim=1))
        return x, rocket_out


class StackingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config['model']
        swin_config = model_config['rd']
        rocket_config = model_config['track']
        self.swin = SwinTransformer3D(**swin_config)
        self.rocket = MultiRocketClassifier(**rocket_config)
        self.num_classes = model_config['num_classes']
        self.model = StackingModel(
            self.swin, self.rocket,
            model_config['hidden_channels'],
            self.num_classes,
            model_config['freeze_stage']
        )

        self.seq_len = config["data"]["track_seq_len"]
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0.
        self.accuracies = torch.zeros(self.seq_len)
        self.count = 0
        self.num_classes = model_config["num_classes"]
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizer = get_optimizer(train_config, self.model)
        scheduler = get_scheduler(train_config, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss"
        }

    def update_conf_matrix(self, predictions, labels):
        for i in range(len(predictions)):
            pred = predictions[i]
            unique_values, counts = torch.unique(pred, return_counts=True)
            pred_label = unique_values[counts.argmax()]
            label = labels[i]
            self.conf_matrix[int(label), int(pred_label)] += 1

    def forward(self, t, sequence: torch.Tensor, last_logits: Optional[torch.Tensor] = None,
                rd_matrices: Optional[torch.Tensor] = None, point_index: Optional[torch.Tensor] = None,
                extra_features: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        sequence_t = sequence[:, :, :t]
        if rd_matrices is None:
            probs, logits = self.model(sequence_t, last_logits=last_logits)
        else:
            index_mask_t = (point_index <= t)
            rd_matrix_t = rd_matrices * index_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_t = mask * index_mask_t if mask is not None else None
            extra_features_t = torch.stack([
                extra_features[i, point_index[i] <= t].mean(0) for i in range(len(extra_features))
            ])
            probs, logits = self.model(sequence_t, rd_matrix_t, extra_features_t, mask_t, last_logits)
        return probs, logits

    def step(self, batch, require_conf=False):
        sequence = batch["sequence"].transpose(1, 2)
        labels = batch["label"]
        last_logits = torch.ones((sequence.shape[0], self.num_classes)).to(sequence) / self.num_classes
        rd_matrices = batch.get("rd_matrices", None)
        point_index = batch.get("point_index", None)
        extra_features = batch.get("extra_features", None)
        mask = batch.get("mask", None)

        b = point_index.shape[0]
        accuracy = torch.zeros(self.seq_len)
        accuracy_by_class = torch.zeros(self.num_classes, self.seq_len)
        begin_time = torch.ones(b) * self.seq_len
        begin = [False for _ in range(b)]
        prediction = torch.zeros((b, self.seq_len))
        total_loss = 0.
        for t in range(1, self.seq_len + 1):
            probs, logits = self.forward(t, sequence, last_logits, rd_matrices, point_index, extra_features, mask)
            last_logits = logits
            loss = self.criterion(probs, labels)
            total_loss += loss

            _, pred = probs.max(1)
            prediction[:, t - 1] = pred
            accuracy[t - 1] = (pred == labels).float().mean().item()
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
        begin_time = result["begin_time"]
        rate = result["rate"]
        accuracy = result["accuracy"]
        accuracy_by_class = result["accuracy_by_class"]
        strict_accuracy = result["strict_accuracy"]
        self.accuracies += b * accuracy
        self.count += b

        self.log_dict({
            "train/loss": self.loss,
            "train/avg_accuracy": accuracy.mean(),
            "train/strict_accuracy": strict_accuracy,
            "train/begin_time": begin_time.mean(),
            "train/rate": rate,
        }, on_step=False, on_epoch=True, batch_size=b)
        for i in range(len(accuracy_by_class)):
            self.log(f"train/acc_cls_{i}", accuracy_by_class[i], on_step=False, on_epoch=True, batch_size=b)

        return self.loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], on_step=False, on_epoch=True)
        self.logger.experiment.add_scalars("train/accuracy",
                                           {f"t{i + 1}": self.accuracies[i] for i in range(0, self.seq_len, 3)},
                                           self.current_epoch)

    def on_validation_epoch_start(self):
        logger.info(f"Start validation stage.")
        self.accuracies = torch.zeros(self.seq_len)
        self.count = 0
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)

    def validation_step(self, batch, batch_idx):
        b = batch["rd_matrices"].shape[0]
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

        return self.loss

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
        b = batch["rd_matrices"].shape[0]
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