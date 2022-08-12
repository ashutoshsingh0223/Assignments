from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import Resize

from fraunhofer.constants import BASE_DIR, BEACH, HARBOR, RIVER


class ClassificationDataset(Dataset):
    CLASS_MAP_BY_INDEX = (BEACH, HARBOR, RIVER, )

    def __init__(self, root: 'Path' = BASE_DIR / 'dataset', train: bool = True, download=False, transform=None,
                 reverse=True):
        self.root = root

        class_0_path = self.root / self.CLASS_MAP_BY_INDEX[0]
        class_1_path = self.root / self.CLASS_MAP_BY_INDEX[1]
        class_2_path = self.root / self.CLASS_MAP_BY_INDEX[2]

        class_0_files = sorted([f for f in class_0_path.glob('**/*')], reverse=reverse)
        class_1_files = sorted([f for f in class_1_path.glob('**/*')], reverse=reverse)
        class_2_files = sorted([f for f in class_2_path.glob('**/*')], reverse=reverse)

        index = len(class_1_files) - len(class_1_files) // 5
        if train:
            self.files = class_0_files[:index] + class_1_files[:index] + class_2_files[:index]
        else:
            self.files = class_0_files[index:] + class_1_files[index:] + class_2_files[index:]

        self.resize_transform = Resize((256, 256))

        self.transform = None
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        label_name = str(file).split('/')[-2]
        label = self.CLASS_MAP_BY_INDEX.index(label_name)
        image = np.array(Image.open(file).convert("RGB")).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = self.resize_transform(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
