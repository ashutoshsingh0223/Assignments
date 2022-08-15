from torch import tensor

from PIL import Image

from helsing.sum.dataset import PairSample


class PairSampleWithOp(PairSample):

    def __init__(self, root: 'Path', train: bool = True, download=True, transform=False):
        super(PairSampleWithOp, self).__init__(root=root, train=train, download=download, transform=transform)

        # Extend the functionality of generating pairs by doubling the size of dataset half for positive op and other
        # half for negative op.
        # Op encoded as one-hot encoding [1., 0.] for positive and [0., 1.] for negative.

        self.plus = 0
        self.minus = 1

        if self.train:
            len_ = len(self.train_pairs)
            self.train_pairs = self.train_pairs + self.train_pairs
            self.ops = ([[1, 0]] * len_) + ([[0, 1]] * len_)
        else:
            len_ = len(self.test_pairs)
            self.test_pairs = self.test_pairs + self.test_pairs
            self.ops = ([[1, 0]] * len_) + ([[0, 1]] * len_)

        self.min = -9

        self.value_to_label = [-9, -8, -7, -6, -5, -4, -3, -2, -1] + list(range(0, 19))

    def __len__(self):
        if self.train:
            return len(self.train_pairs)
        else:
            return len(self.test_pairs)

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

        op = self.ops[index]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if op[self.plus] == 1:
            val = label1 + label2
            label = tensor(val + 9)
        elif op[self.minus] == 1:
            val = label1 - label2
            label = tensor(self.value_to_label.index(val))

        return self.to_tensor(img1), self.to_tensor(img2), tensor(op), label


class PairSampleWithOpAndSign(PairSampleWithOp):
    def __init__(self, root: 'Path', train: bool = True, download=True, transform=False):
        super(PairSampleWithOpAndSign, self).__init__(root=root, train=train, download=download, transform=transform)

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

        op = self.ops[index]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        sign = 1
        if op[self.plus] == 1:
            val = label1 + label2
            label = tensor(val)
        elif op[self.minus] == 1:
            val = label1 - label2
            label = tensor(abs(val))
            if val < 0:
                sign = 0

        sign = tensor(sign)
        regression_target = (val - self.min) / (self.max - self.min)
        return self.to_tensor(img1), self.to_tensor(img2), tensor(op), label, sign, regression_target
