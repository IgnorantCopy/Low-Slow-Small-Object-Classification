import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from einops import rearrange
from thop import profile
from timm.layers import DropPath, trunc_normal_
from functools import lru_cache
from typing import Optional

from src.low_slow_small_object_classification.utils.logger import default_logger as logger
from src.low_slow_small_object_classification.utils.config import get_optimizer, get_scheduler
from src.low_slow_small_object_classification.models.utils import calc_rate


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, activation=nn.GELU, dropout=0.):
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def window_partition(x: torch.Tensor, window_size: tuple[int, int, int]):
    x = rearrange(x, "b (d_w w1) (h_w w2) (w_w w3) c -> (b d_w h_w w_w) (w1 w2 w3) c", w1=window_size[0],
                  w2=window_size[1], w3=window_size[2])
    return x


def window_reverse(windows, window_size: tuple[int, int, int], d: int, h: int, w: int):
    w1, w2, w3 = window_size
    d_w, h_w, w_w = d // w1, h // w2, w // w3
    x = rearrange(windows, "(b d_w h_w w_w) w1 w2 w3 c -> b (d_w w1) (h_w w2) (w_w w3) c", d_w=d_w, h_w=h_w, w_w=w_w)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, heads, qkv_bias=False, scale=None, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size, window_size) if isinstance(window_size, int) else window_size
        self.heads = heads
        self.qkv_bias = qkv_bias
        self.scale = scale or (dim // heads) ** -0.5

        self.window_d, self.window_h, self.window_w = self.window_size
        # relative position bias table
        self.pbt = nn.Parameter(
            torch.zeros((2 * self.window_d - 1) * (2 * self.window_h - 1) * (2 * self.window_w - 1), heads)
        )
        # relative position index
        coords_d = torch.arange(self.window_d)
        coords_h = torch.arange(self.window_h)
        coords_w = torch.arange(self.window_w)
        coords = torch.stack(
            torch.meshgrid([coords_d, coords_h, coords_w])
        ).flatten(start_dim=1)  # [3, window_d * window_h * window_w]
        relative_coords = coords[:, :, None] - coords[:, None, :]  # [3, w_d * w_h * w_w, w_d * w_h * w_w]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [w_d * w_h * w_w, w_d * w_h * w_w, 3]
        relative_coords[:, :, 0] += self.window_d - 1
        relative_coords[:, :, 1] += self.window_h - 1
        relative_coords[:, :, 2] += self.window_w - 1

        relative_coords[:, :, 0] *= (2 * self.window_h - 1) * (2 * self.window_w - 1)
        relative_coords[:, :, 1] *= 2 * self.window_w - 1
        relative_position_index = relative_coords.sum(-1)  # [w_d * w_h * w_w, w_d * w_h * w_w]
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.pbt, std=0.02)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        attention = self.scale * q @ k.transpose(-2, -1)

        relative_cpb = self.pbt[
            self.relative_position_index[:n, :n].reshape(-1)
        ].reshape(n, n, -1).permute(2, 0, 1).contiguous()  # [w_d * w_h * w_w, w_d * w_h * w_w, heads]
        attention += relative_cpb.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attention = attention.view(b // n_w, n_w, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.heads, n, n)
        attention = self.attn_dropout(self.softmax(attention))
        x = (attention @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_dropout(self.proj(x))


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, heads, window_size=(2, 7, 7), shift_size=(0, 0, 0), ff_ratio=4., qkv_bias=True,
                 scale=None, dropout=0., attn_dropout=0., dropout_path=0., activation=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.ff_ratio = ff_ratio

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm(dim)
        self.attention = WindowAttention3D(dim, window_size, heads, qkv_bias, scale, attn_dropout, proj_dropout=dropout)
        self.drop_path = DropPath(dropout_path) if dropout_path > 0 else nn.Identity()
        self.norm2 = norm(dim)
        self.ff = FeedForward(dim, hidden_channels=int(dim * ff_ratio), activation=activation, dropout=dropout)

    def forward(self, x, mask):
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        w_d, w_h, w_w = window_size
        s_d, s_h, s_w = shift_size

        shortcut = x
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (w_d - d % w_d) % w_d
        pad_b = (w_h - h % w_h) % w_h
        pad_r = (w_w - w % w_w) % w_w
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, d_p, h_p, w_p, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-s_d, -s_w, -s_h), dims=(1, 2, 3))
            attn_mask = mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)
        # W-MSA/SW-MSA
        attn_windows = self.attention(x_windows, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, d, h, w)
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(s_d, s_w, s_h), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)

        return x + self.drop_path(self.ff(self.norm2(x)))


class PatchMerge(nn.Module):
    def __init__(self, dim, norm=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm(2 * dim)

    def forward(self, x):
        b, d, h, w, c = x.shape
        pad_input = (h % 2 == 1) or (w % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [b, d, h/2 * w/2, 4 * c]

        return self.norm(self.reduction(x))


# cache the window partition and window reverse functions for faster computation
@lru_cache()
def compute_mask(d, h, w, window_size, shift_size, device, external_mask=None):
    w_d, w_h, w_w = window_size
    s_d, s_h, s_w = shift_size
    image_mask = torch.zeros((1, d, h, w, 1), device=device)
    count = 0
    for dd in slice(-w_d), slice(-w_d, -s_d), slice(-s_d, None):
        for hh in slice(-w_h), slice(-w_h, s_h), slice(s_h, None):
            for ww in slice(-w_w), slice(-w_w, s_w), slice(s_w, None):
                if external_mask is not None:
                    effective_mask = external_mask[:, dd, hh, ww, :]
                    if effective_mask.sum() > 0:
                        image_mask[:, dd, hh, ww, :] = count
                        count += 1
                else:
                    image_mask[:, dd, hh, ww, :] = count
                    count += 1
    mask_windows = window_partition(image_mask, window_size).squeeze(-1)  # [w, w_d * w_h * w_w]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, heads, window_size, ff_ratio=4., qkv_bias=True, scale=None,
                 dropout=0., attn_dropout=0., dropout_path=0., norm=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.window_size = (window_size, window_size, window_size) if isinstance(window_size, int) else window_size
        self.shift_size = tuple(i // 2 for i in window_size)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim, heads=heads, window_size=window_size, ff_ratio=ff_ratio, norm=norm,
                                   shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                                   qkv_bias=qkv_bias, scale=scale, dropout=dropout, attn_dropout=attn_dropout,
                                   dropout_path=dropout_path[i] if isinstance(dropout_path, list) else dropout_path)
            for i in range(depth)
        ])
        self.pool = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)
        self.downsample = downsample(dim=dim, norm=norm) if downsample else None

    def forward(self, x, mask=None):
        b, c, d, h, w = x.shape
        shortcut = rearrange(self.pool(rearrange(x, 'b c d h w -> (b d) c h w')), '(b d) c h w -> b c d h w', b=b) \
            if self.downsample else x
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        w_d, w_h, w_w = window_size
        x = rearrange(x, 'b c d h w -> b d h w c')
        p_d = int(np.ceil(d / w_d)) * w_d
        p_h = int(np.ceil(h / w_h)) * w_h
        p_w = int(np.ceil(w / w_w)) * w_w

        if mask is not None:
            t = mask.shape[0]
            mask = mask.reshape(d, -1).sum(1)
            mask = (mask >= t // d // 2).int()
            mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((1, d, h, w, 1))

        attn_mask = compute_mask(p_d, p_h, p_w, window_size, shift_size, x.device, mask)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x + shortcut


class PatchEmbedding3D(nn.Module):
    def __init__(self, patch_size, in_channels=3, embed_dim=96, norm=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm(embed_dim) if norm else None

    def forward(self, x):
        x = rearrange(x, 'b d c h w -> b c d h w')
        _, _, d, h, w = x.shape
        p_d, p_h, p_w = self.patch_size
        if w % p_w != 0:
            x = F.pad(x, (0, p_w - w % p_w))
        if h % p_h != 0:
            x = F.pad(x, (0, 0, 0, p_h - h % p_h))
        if d % p_d != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, p_d - d % p_d))

        x = self.proj(x)  # [b, c, d, w_h, w_w]

        if self.norm:
            _, _, d, w_h, w_w = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, d, w_h, w_w)
        return x


class SwinTransformer3D(nn.Module):
    def __init__(self, patch_size, num_classes, extra_features_dim=0, in_channels=3, embed_dim=96, depths=None,
                 heads=None, window_size=(2, 7, 7), ff_ratio=4., qkv_bias=True, scale=None, dropout=0., attn_dropout=0.,
                 dropout_path=0.1, norm="LayerNorm", patch_norm=False, frozen_stages=-1):
        super().__init__()
        if heads is None:
            heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.window_size = window_size
        self.frozen_stages = frozen_stages

        if norm == "LayerNorm":
            norm = nn.LayerNorm
        elif norm == "BatchNorm":
            norm =nn.BatchNorm3d
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        self.patch_embedding = PatchEmbedding3D(
            patch_size, in_channels, embed_dim, norm if self.patch_norm else None,
        )

        self.pos_dropout = nn.Dropout(dropout)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, dropout_path, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i), depth=depths[i], heads=heads[i], window_size=window_size,
                ff_ratio=ff_ratio, qkv_bias=qkv_bias, scale=scale, dropout=dropout, attn_dropout=attn_dropout,
                dropout_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm=norm,
                downsample=PatchMerge if i < self.num_layers - 1 else None,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm(self.num_features)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features + extra_features_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embedding.eval()
            for param in self.patch_embedding.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1:
            self.pos_dropout.eval()
            for i in range(self.frozen_stages):
                module = self.layers[i]
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, extra_features=None, mask=None):
        x = self.patch_embedding(x)
        x = self.pos_dropout(x)
        for layer in self.layers:
            x = layer(x.contiguous(), mask)

        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c (d h w)')
        x = self.avg_pool(x).flatten(1)
        if extra_features is not None:
            x = torch.cat([x, extra_features], dim=1)
        x = self.softmax(self.head(x))
        return x

    def train(self, mode=True):
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()


class Swin3D(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config["model"]
        model_config.pop("name", None)
        self.model = SwinTransformer3D(**model_config)
        self.automatic_optimization = False

        self.seq_len = config["data"]["track_seq_len"]
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0.
        self.weights = torch.tensor([self.seq_len - i for i in range(self.seq_len)])
        self.weights = self.weights / self.weights.sum()
        self.accuracies = torch.zeros(self.seq_len)
        self.count = 0
        self.num_classes = model_config["num_classes"]
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizer = get_optimizer(train_config, self.model)
        scheduler = get_scheduler(train_config, optimizer)
        return [optimizer], [scheduler]

    def update_conf_matrix(self, predictions, labels):
        for i in range(len(predictions)):
            pred = predictions[i]
            unique_values, counts = torch.unique(pred, return_counts=True)
            pred_label = unique_values[counts.argmax()]
            label = labels[i]
            self.conf_matrix[int(label), int(pred_label)] += 1

    def forward(self, t, point_index: torch.Tensor, rd_matrices: torch.Tensor,
                extra_features: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        index_mask_t = (point_index <= t)
        rd_matrix_t = rd_matrices * index_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask_t = mask * index_mask_t if mask is not None else None
        extra_features_t = torch.stack([
            extra_features[i, point_index[i] <= t].mean(0) for i in range(len(extra_features))
        ])

        logits = self.model(rd_matrix_t, extra_features_t, mask_t)
        return logits

    def step(self, batch, require_conf=False):
        rd_matrices = batch["rd_matrices"]
        point_index = batch["point_index"]
        extra_features = batch["extra_features"]
        mask = batch["mask"]
        labels = batch["label"]

        b = point_index.shape[0]
        accuracy = torch.zeros(self.seq_len)
        accuracy_by_class = torch.zeros(self.num_classes, self.seq_len)
        begin_time = torch.ones(b) * self.seq_len
        begin = [False for _ in range(b)]
        prediction = torch.zeros((b, self.seq_len))
        total_loss = 0.
        for t in range(1, self.seq_len + 1):
            logits = self.forward(t, point_index, rd_matrices, extra_features, mask)
            loss = self.weights[t - 1] * self.criterion(logits, labels)
            total_loss += loss

            _, pred = logits.max(1)
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

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(self.loss)
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
        scheduler = self.lr_schedulers()
        scheduler.step()
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
