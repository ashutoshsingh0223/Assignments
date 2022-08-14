from PIL import Image

from helsing.sum.dataset import PairSample


class PairSampleWithOp(PairSample):

    def __init__(self, root: 'Path', train: bool = True, download=True, transform=False):
        super(PairSampleWithOp, self).__init__(root=root, train=train, download=download, transform=transform)

        # Extend the functionality of generating pairs by doubling the size of dataset half for positive op and other
        # half for negative op.
        # Op encoded as one-hot encoding [1., 0.] for positive and [0., 1.] for negative.
        if self.train:
            self.train_pairs = self.train_pairs + self.train_pairs

        else:
            self.test_pairs = self.test_pairs + self.test_pairs

    def __get_item__(self, index):
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


