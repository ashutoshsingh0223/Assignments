# Coding challenge

_Fraunhofer IGD - Computer Vision_

_Image classification with PyTorch_

## Task

Imagine participating in a research project and being assigned the task of implementing and training a neural network. 

The neural network you are training is supposed to classify image data. For the training, you will be provided with images that belong to one of the following three classes.

* river
* harbour
* beach

After completing the implementation and training, we will discuss your code together. In total, this task should not take more than 2 to 3 hours of your time. 

## Implementation

The context of the project is that a Convolutional Neural Network will be used to classify the data. The implementation will be done with Python, PyTorch and Torchvision.

Also, the use of the following libraries is allowed:

* numpy
* pandas
* matplotlib
* opencv-python
* Pillow
* kornia

The end result should be two Python scripts:

* train.py containing all the training code
* test.py which can be used to test the trained network with an arbitrary image file

We ask that you do not use any pre-built models from Torchvision. Instead, you should define a simple CNN architecture yourself using the building blocks available in Torchvision.

Except for these defaults, you are free to design the code as you wish. As in a real research project, the aim of this task is to achieve the highest possible performance of the trained model. However, the quality of the code also plays a role!

## Data

The images are located in the compressed directory _dataset_. The subdirectories were named according to the classes of the images.  

## Timeline

We assume that you should solve this task in a time frame of 1 week. If you need longer, please contact us about it. 

## Notes

It's fine if you don't have access to GPUs for training. The dataset is small enough and the task is comparatively simple so that a good network can be trained in as little as 10 minutes using a notebook CPU.


# Solution

## Dataset structure
Please put your dataset inside the `fraunhofer` directory. The directory structure classes should look like `<path>/fraunhofer/dataset/<class>/*.png` , for example `<path>/fraunhofer/dataset/beach/*.png`

## Project Structure - fraunhofer

### Model File
Contains all the information about the model and encoder and examples about how their configs are parametrized.

### Dataset File
Torch dataset class to load and get the dataset. 20% dataset isn used for validation.

### Utils
Some utils for metric calculation and plotting etc.

## Train and Test Scripts

### train.py
Used to train a classifier with predefined config and number of classes, uses adam optimizer and cross entropy loss for optimization.
This file is not parametrized and runs for batch size of 32 for 15 epochs. It is also responsible for saving and plotting metrics and other artifacts.

```
python train.py
```

#### Saved models after training
Each training iteration is linked to a an indentifier called the `run_id`. 
After training is complete a new directory is created like `<path>/fraunhofer/run_id` which contains the folowwing files:

- `best-classifier.pt`: Saved model state dict for best model version
- `classifier.pt`: Saved model state dict for final model
- `hyperparams.json`: All the configs & hyperparams used for training.
- `metrics.json`: Metrics per epoch; accuracy, loss, f1_score, and roc-auc
- `val_metrics.png`: Plots for validation metrics


### test.py
Runs one-time or interactive predictions using `single` and `multiple` prediction modes. How to run?

Prediction mode `single` takes in a saved model path and an image path to spit out predicted class(top-1) for the image.
```
python test.py --model-path /1/2/3/best-classifier.pt --prediction-mode single --img-path /1/2/3/fraunhofer/dataset/beach/19.png
```

Output:

```
/1/2/3/fraunhofer/dataset/beach/19.png
Predicted Class: beach
```


Prediction mode `multiple` does the same thing, but you can interactively provide image paths. Example:

```
python test.py --model-path /1/2/3/best-classifier.pt --prediction-mode multiple
```

Output:

```
Enter Image : /1/2/3/fraunhofer/dataset/beach/19.png
/1/2/3/fraunhofer/dataset/beach/19.png
Predicted Class: beach

Enter Image Path: /1/2/3/fraunhofer/dataset/beach/22.png        
/1/2/3/fraunhofer/dataset/beach/22.png 
Predicted Class: beach

Enter Image Path: /1/2/3/fraunhofer/dataset/beach/22.png        
/1/2/3/fraunhofer/dataset/beach/22.png 
Predicted Class: beach

```
