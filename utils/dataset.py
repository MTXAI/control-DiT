import os

import numpy as np
import torch
from torchvision.datasets import ImageFolder


class CustomDataset(ImageFolder):
    def __init__(self, root: str, features_dir, labels_dir, conditions_dir, transform=None):
        super().__init__(root, transform)
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.conditions_dir = conditions_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.conditions_files = sorted(os.listdir(conditions_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files) == len(self.conditions_files), \
            "Number of feature files, label files and condition files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]
        condition_file = self.conditions_files[idx]

        feature = np.load(os.path.join(self.features_dir, feature_file))
        label = np.load(os.path.join(self.labels_dir, label_file))
        condition = np.load(os.path.join(self.conditions_dir, condition_file))
        print(torch.from_numpy(condition).shape)
        return torch.from_numpy(feature), torch.from_numpy(label), torch.from_numpy(condition)
