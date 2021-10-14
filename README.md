# ImageDenoisingAutoencoder

The Denoising autoencoder is implemented with Pytorch and is applied on the MNIST dataset.


## Table of contents

 - [Required Imports](#required-imports)
 - [Dataset](#dataset)
 - [Denoising Autoencoder](#denoising-autoencoder)
 - [Model architecture](#model-architecture)
 - [Hyperparameters](#hyperparameters)
 - [Training](#training)
 - [Training-Loss plot](#training-loss-plot)   
## Required Imports

* numpy
* matplotlib(for data visualization)
* Pytorch(for building and training the autoencoder)
## Dataset
The [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) is used. It is a dataset of 60,000 training samples and 10,000 test samples. Each sample is a 28×28 pixel grayscale image of a single handwritten digit between 0 and 9.

## Denoising Autoencoder
* Components of Autoencoder:
    * Encoder - This network downsamples the data into lower dimensions.
    * Decoder - This network reconstructs the original data from the lower dimensional representation
* In denoising autoencoders some noise is introduced to the input images. It tries to reconstruct the original image without noise. Such a noisy input reduces the risk of overfitting and prevents the autoencoder from learning a simple identity function.

## Model architecture
```python:
#Encoder
      nn.Conv2d(1,16,3, stride = 2, padding = 1)
      nn.ReLU(),
      nn.Conv2d(16,32,3, stride = 2, padding = 1)
      nn.ReLU()
      nn.Conv2d(32,64,7)
#Decoder
      nn.ConvTranspose2d(64,32,7)
      nn.ReLU()
      nn.ConvTranspose2d(32,16,3, stride = 2, padding=1, output_padding = 1)
      nn.ReLU()
      nn.ConvTranspose2d(16,1,3, stride = 2, padding=1, output_padding = 1)
      nn.Sigmoid()    
```

## Hyperparameters

|Hyperparameter     |Value |
| :----------- | :----------- |
| Learning rate      | 0.001      |
| Number of epochs   | 10       |
| Batch Size     | 64      |

## Training

#### Adding noise
A function is defined to add noise to the images. `torch.randn` is used to create a noisy tensor of the same size as the input.
#### Optimizer and Loss function
MSE loss and the Adam optimization technique is used.

## Training-Loss plot


