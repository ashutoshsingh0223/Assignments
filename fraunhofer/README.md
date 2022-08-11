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



