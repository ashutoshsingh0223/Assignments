import torch
from torch.utils.data import Dataset

from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from PIL import Image

import numpy as np


class PairSample(Dataset):
    def __init__(self, root: 'Path', train: bool = True, download=True, transform=False):

        self.root = root

        self.to_tensor = ToTensor()

        self.transform = None
        if transform:
            mean, std = 0.1307, 0.3081
            self.transform = Compose([ToTensor(), Normalize((mean,), (std,))])

        train_dataset = MNIST(self.root / 'MNIST', train=True, download=download,
                              transform=self.transform)
        test_dataset = MNIST(self.root / 'MNIST', train=False, download=download,
                             transform=self.transform)

        self.n_classes = 19

        self.train = train
        if self.train:
            self.mnist_dataset = train_dataset
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            self.train_pairs = []
            for i in range(0, len(self.train_data)):
                sampled_label = np.random.choice(list(self.labels_set))
                sampled_index = np.random.choice(self.label_to_indices[sampled_label])
                self.train_pairs.append((i, sampled_index))

        else:
            # generate fixed pairs for testing
            self.mnist_dataset = test_dataset
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            self.test_pairs = []
            for i in range(0, len(self.test_data)):
                sampled_label = random_state.choice(list(self.labels_set))
                sampled_index = random_state.choice(self.label_to_indices[sampled_label])
                self.test_pairs.append((i, sampled_index))

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[self.train_pairs[index][0]], self.train_labels[
                self.train_pairs[index][0]].item()
            img2, label2 = self.train_data[self.train_pairs[index][1]], self.train_labels[
                self.train_pairs[index][1]].item()
        else:
            img1, label1 = self.test_data[self.test_pairs[index][0]], self.test_labels[
                self.test_pairs[index][0]].item()
            img2, label2 = self.test_data[self.test_pairs[index][1]], self.test_labels[
                self.test_pairs[index][1]].item()

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        return self.to_tensor(img1), self.to_tensor(img2), torch.tensor(label1+label2)
