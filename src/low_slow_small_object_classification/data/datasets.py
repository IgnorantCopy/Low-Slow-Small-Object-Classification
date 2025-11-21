import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as pl
from sklearn.model_selection import train_test_split
from typing import List

from src.low_slow_small_object_classification.data.rd_preprocessor import BatchFile, process_batch
from src.low_slow_small_object_classification.data.track_preprocessor import TrajectoryPreprocessor
from src.low_slow_small_object_classification.data.utils import get_batch_file_list

# region RDMap

class RDMap(Dataset):
    def __init__(self, batch_files: List[BatchFile], transform=None, seq_len=180):
        super().__init__()

        self.transform = transform
        self.seq_len = seq_len
        self.batch_files = batch_files

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        batch_file = self.batch_files[idx]
        label = batch_file.label - 1
        track_file = batch_file.track_file

        batch_info = process_batch(batch_file)
        if batch_info is None:
            return None
        rd_matrices = batch_info["rd_matrices"]
        point_index = torch.from_numpy(batch_info["point_index"])
        extra_features = torch.from_numpy(batch_info["extra_features"])

        flip = np.random.rand() > 0.5
        for i in range(len(rd_matrices)):
            if self.transform is not None:
                rd_matrices[i] = self.transform(rd_matrices[i])
                if flip:
                    rd_matrices[i] = torch.flip(rd_matrices[i], [2])
        rd_matrices = torch.stack(rd_matrices).float()

        mask = torch.ones((self.seq_len,), dtype=torch.int32)
        if len(rd_matrices) < self.seq_len:
            mask[len(rd_matrices):] = 0
            diff = self.seq_len - len(rd_matrices)
            rd_matrices = torch.cat([rd_matrices, torch.zeros((diff, *rd_matrices.shape[1:]))], dim=0)
            point_index = torch.cat([point_index, torch.tensor([point_index[-1] for _ in range(diff)])], dim=0)
            extra_features = torch.cat([extra_features, *[extra_features[-1:, :] for _ in range(diff)]], dim=0)
        elif len(rd_matrices) > self.seq_len:
            quantiles = np.linspace(0, 1, self.seq_len)
            indices = np.round(quantiles * (len(rd_matrices) - 1)).astype(np.int32)
            rd_matrices = rd_matrices[indices]
            point_index = point_index[indices]
            extra_features = extra_features[indices]

        return {
            "rd_matrices": rd_matrices,
            "point_index": point_index,
            "extra_features": extra_features,
            "mask": mask,
            "label": label,
            "track_file": track_file
        }


class RDDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()

        self.data_root = data_config["data_root"]
        self.val_ratio = data_config["val_ratio"]
        self.shuffle = data_config.get("shuffle", True)
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config["num_workers"]
        self.seq_len = data_config["image_seq_len"]
        self.height = data_config["height"]
        self.width = data_config["width"]
        self.batch_files = get_batch_file_list(self.data_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.height, self.width)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def setup(self, stage=None):
        train_batch_files, val_batch_files = train_test_split(self.batch_files, test_size=self.val_ratio)
        self.train_dataset = RDMap(train_batch_files, transform=self.transform, seq_len=self.seq_len)
        self.val_dataset = RDMap(val_batch_files, transform=self.transform, seq_len=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# endregion

# region TrackData

class TrajectoryDataset(Dataset):
    def __init__(self, batch_files: List[BatchFile], seq_len: int = 20):
        super().__init__()

        self.batch_files = batch_files
        self.preprocessor = TrajectoryPreprocessor(seq_len)

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        batch_file = self.batch_files[idx]
        return self.preprocessor.process_single_trajectory(batch_file)


class TrackDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()

        self.data_root = data_config["data_root"]
        self.val_ratio = data_config["val_ratio"]
        self.shuffle = data_config.get("shuffle", True)
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config["num_workers"]
        self.seq_len = data_config["track_seq_len"]
        self.batch_files = get_batch_file_list(self.data_root)

    def setup(self, stage=None):
        train_batch_files, val_batch_files = train_test_split(self.batch_files, test_size=self.val_ratio)
        self.train_dataset = TrajectoryDataset(train_batch_files, seq_len=self.seq_len)
        self.val_dataset = TrajectoryDataset(val_batch_files, seq_len=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# endregion

# region MultiDataset

class MultiDataset(Dataset):
    def __init__(self, rd_dataset: RDMap, track_dataset: TrajectoryDataset):
        super().__init__()

        self.rd_dataset = rd_dataset
        self.track_dataset = track_dataset

    def __len__(self):
        return len(self.rd_dataset)

    def __getitem__(self, idx):
        rd_data = self.rd_dataset[idx]
        if rd_data is None:
            rd_data = dict()
        track_data = self.track_dataset[idx]
        rd_data.update(track_data)
        return rd_data


class MultiDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()

        self.shuffle = data_config.get("shuffle", True)
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config["num_workers"]
        self.rd_datamodule = RDDataModule(data_config)
        self.track_datamodule = TrackDataModule(data_config)

    def setup(self, stage=None):
        self.rd_datamodule.setup(stage)
        self.track_datamodule.setup(stage)
        self.train_dataset = MultiDataset(self.rd_datamodule.train_dataset, self.track_datamodule.train_dataset)
        self.val_dataset = MultiDataset(self.rd_datamodule.val_dataset, self.track_datamodule.val_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)