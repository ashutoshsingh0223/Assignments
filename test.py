import torch
from torchvision.transforms import Resize
from torch.nn import Softmax
from torch import from_numpy, device, cuda

import numpy as np
from PIL import Image

from fraunhofer.model import Classifier

def predict(model_path, img_path):
    transform = Resize((256, 256))
    image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = from_numpy(image)
    image = transform(image)
    # Convert to a batch of 1
    image = image.unsqueeze(0)

    dev = device("cuda:0" if cuda.is_available() else "cpu")

    model = Classifier.load(model_path)
    model = model.to(dev)
    image = image.to(device)

    softmax = Softmax(dim=1)

    logits = model(image)
    preds = softmax(logits)

    top_1_class = torch.argmax(preds, dim=1)
    print(top_1_class)

