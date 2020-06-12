# IMSC-ACM
Classification system as a part of ACM Multimedia Product Image Recognition Challenge using PyTorch and transfer learning techniques

**Problem Statment:** Find exact match for test images from training image dataset with 0.5 million beauty product images 

**Network Architecture:** ResNet based autoencoder pretrained on ImageNet.
The code downloads ResNet architecture pretrained on imageNet from pyTorch.org or loads the model if it is already downloaded.

- Autoencoder has two parts an encoder which reduces the feature dimension and a decoder which tries to reconstruct the image using the reduced features. 
- The input and ground truth for training the model are both the product images, hence the dataloader is written accordingly.
- logging module is used to record the training loss and accuracy of the model into a log file.
- tqdm module is used to get a smart progress bar during the training which helps visualization.
- Trained models are saved after every 50 epochs into the output folder specified.


### Requirements:
###### Linux development environment. You will need the following to run the scripts:
```
PyTorch and Python 3 version
Torch, torchvision
Numpy
Glob
PIL
tqdm
logging
GPU and required NVIDIA software to run pyTorch on a GPU (cuda, etc)

```

