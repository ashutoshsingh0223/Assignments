import torch
from torchvision.transforms import Resize
from torch.nn import Softmax
from torch import from_numpy, device, cuda, set_grad_enabled

import numpy as np
from PIL import Image

from fraunhofer.model import Classifier
from fraunhofer.dataset import ClassificationDataset


def load_image(img_path):
    transform = Resize((256, 256))
    image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = from_numpy(image)
    image = transform(image)
    # Convert to a batch of 1
    image = image.unsqueeze(0)
    return image


def load_model(model_path):
    model = Classifier.load(model_path)
    return model


def predict(model, image):
    dev = device("cuda:0" if cuda.is_available() else "cpu")

    model.eval()
    with set_grad_enabled(False):
        model = model.to(dev)
        image = image.to(dev)

        softmax = Softmax(dim=1)

        logits = model(image)
        preds = softmax(logits)
        top_1_class = torch.argmax(preds, dim=1)

    class_id = ClassificationDataset.CLASS_MAP_BY_INDEX[top_1_class[0].cpu().item()]
    print(f'Predicted Class: {class_id}')
    return class_id


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model-path', required=True, type=str)
    parser.add_argument('prediction-mode', required=False, type=str, default='single', choices=['single', 'multiple'])
    parser.add_argument('img-path', required=False, type=str, default=None)
    args = parser.parse_args()

    model = load_model(args.model_path)

    if args.prediction_mode == 'single':
        if args.img_path is None:
            raise ValueError('Prediction Model `single` requires image path')
        print(args.img_path)
        image = load_image(args.img_path)
        predict(model, image)
        print('\n')

    elif args.prediction_model == 'multiple':
        img_path = input('Enter Image Path')
        print(img_path)
        image = load_image(img_path)
        predict(model, image)
        print('\n')
