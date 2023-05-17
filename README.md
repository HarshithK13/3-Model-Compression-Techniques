# 3-Model-Compression-Techniques

## Introduction

Machine learning and deep learning have come a long way in the past few years. Each year, both
the accuracy and the size of the models continued to increase. The main problem with using AI
that is up-to-date today is that edge devices don't have a lot of resources. Effective deep learning
models are large in size. But the bigger the model, the more storage space it needs, which makes
it hard to use on devices with few resources. Also, the inference process takes longer and uses
more energy when the model is larger. Even though these models have done well in the lab, they
aren't very useful in the real world. Model compression makes a neural network (NN) smaller
without changing how well it works. This size reduction is important because it's hard to use
bigger NNs on devices with limited resources.

In this project, we have decided to look at the pros and cons of three common techniques used to
compress large models into smaller ones. The three techniques are as follows:

1. Pruning
2. Quantization
3. Knowledge distillation

## Description of Datasets

_**CiFAR10 Dataset**_: The CiFAR10 dataset consists of 60,000 32x32 color images divided into 10
classes, with each class containing 6,000 images. The images in CIFAR-10 are relatively low
resolution (32 x 32 pixels), but they are still challenging for image classification tasks. The
dataset is well-suited for training and evaluating models for object recognition and image
classification.

![image](https://github.com/HarshithK13/3-Model-Compression-Techniques/assets/84466567/a29e5cee-705c-4410-907b-81a969a670bf)

_**The Street View House Numbers (SVHN) dataset**_: The SVHN dataset is a large collection of
images of numbers that has been used a lot in computer vision research. The dataset is made up
of more than 600,000 images of house numbers taken from Google Street View. There are 10
different types of digits (0–9) in the dataset.

![image](https://github.com/HarshithK13/3-Model-Compression-Techniques/assets/84466567/f5a4bda5-6297-4079-9891-f840f1021cdc)    

We applied the three different model compression techniques on both datasets and then compared how
well they did on each dataset to find the best algorithm for that dataset. The following models were used
on data sets from different fields:

_**DenseNet-121**_: The DenseNet-121 architecture is based on the idea of "Dense Blocks," which
are made up of many convolutional layers that are densely connected to each other. This means
that each layer gets information from both the layer before it in the block and the layer before it
in the network. One benefit of the DenseNet-121 architecture is that it needs fewer parameters
than some other popular deep learning architectures and can work well when resources are
limited.

_**ResNet-50**_: It consists of 50 layers, including convolutional layers, pooling layers, fully
connected layers, and shortcut connections. The key innovation in ResNet-50 is the use of
residual blocks, which allow for the training of very deep neural networks by addressing the
problem of vanishing gradients. The residual blocks contain skip connections (shortcut
connections) that bypass one or more convolutional layers, allowing the gradient to flow directly
from earlier layers to subsequent layers.

_**ResNet-34**_: ResNet-34 is a variant of the ResNet architecture with 34 layers. It follows a similar
design principle as ResNet-50, utilizing residual blocks with shortcut connections to enable the
training of deeper networks. The architecture of ResNet-34 consists of convolutional layers,
pooling layers, fully connected layers, and residual blocks, but with fewer layers compared to
ResNet-50.

_**VGG16**_: It consists of 16 layers, including convolutional layers with small filter sizes (3x3),
pooling layers, and fully connected layers. VGG16 has a straightforward and uniform
architecture, with multiple convolutional layers followed by pooling layers, leading to a gradual
reduction in spatial dimensions. Although VGG16 has a larger number of parameters compared
to some other architectures, it has shown strong performance on various image classification
tasks.

We used the pre-trained model for each of the models described above and applied the model
compression techniques to them.

## Model Compression Techniques and Analysis

1. Pruning: Pruning is a powerful way to cut down on the number of parameters in deep
neural networks. In DNNs, many of the parameters are useless because they don't add
much to the training process. So, once the network has been trained, these parameters can
be removed with little effect on accuracy. Pruning makes models smaller, which makes
them run faster. We successfully pruned all models by 20%!

2. Quantization: Quantization makes the original network smaller by cutting down on the
number of bits each weight needs. For example, the weights can have 16-bit, 8-bit, 4-bit,
or even 1-bit quantization. By cutting down on the number of bits used, the DNN can be
made much smaller. Comparing sizes of the models (in MB) after quantization,

3. Knowledge Distillation: In knowledge distillation, a large model is trained using a large
set of data. When this model can work well on data it hasn't seen before and generalize,
the knowledge is passed to a smaller network. The large model is called the teacher
model, and the smaller one is called the student network.

![image](https://github.com/HarshithK13/3-Model-Compression-Techniques/assets/84466567/f529af7e-1e98-4121-9658-b4d50ed4c1a1)

We have trained the two datasets on the four models, which are the teachers. We got the
following, which depict the **Accuracy vs Epoch** plots of each model: (VGG16 - top left,
Resnet50 - top right, Resnet34 - bottom left, Densenet121 - bottom right)


**SVHN Dataset**

![image](https://github.com/HarshithK13/3-Model-Compression-Techniques/assets/84466567/d5d338b3-8822-47af-9bee-e5c133d0d6d9)

**CIFAR10 Dataset**

![image](https://github.com/HarshithK13/3-Model-Compression-Techniques/assets/84466567/c70762e1-8b03-4103-a5cc-0782d9a839bd)

## Project Summary

This project focuses on exploring model compression techniques to reduce the size of deep
neural networks (DNNs) while maintaining their performance. The motivation behind model
compression is the need to deploy DNNs on resource-constrained devices. Four common
compression techniques, namely pruning, quantization, and knowledge distillation, are examined
to evaluate their effectiveness on two datasets: CIFAR10 and Street View House Numbers
(SVHN).

The CIFAR10 dataset consists of 60,000 low-resolution (32x32 pixels) color images categorized
into 10 classes, making it suitable for image classification tasks. On the other hand, the SVHN
dataset contains over 600,000 images of house numbers captured from Google Street View, with
10 different types of digits (0-9).

To assess the performance of the compression techniques, four pre-trained models are employed:
DenseNet-121, ResNet-50, ResNet-34, and VGG16. These models possess different architectures
and have been proven effective in various image classification tasks.

Pruning is the first compression technique examined. It involves removing unnecessary
parameters from a trained DNN without significantly impacting accuracy. By eliminating
redundant parameters, the model becomes smaller and more efficient in terms of inference speed
and energy consumption.

Quantization, the second technique, reduces the memory requirements of the model by reducing
the number of bits used to represent each weight. This can be achieved by quantizing weights to

16-bit, 8-bit, 4-bit, or even 1-bit precision. By reducing the number of bits used, the model size is
significantly reduced.

The third technique, knowledge distillation, involves training a large teacher model on a large
dataset and transferring its knowledge to a smaller student network. The teacher model's ability
to generalize to unseen data is leveraged to enhance the performance of the student network.
The project evaluates and compares the performance of these compression techniques on the
CIFAR10 and SVHN datasets. Accuracy versus epoch plots are obtained for each model to
analyze their performance. These plots provide insights into the trade-off between model size
reduction and accuracy.

In summary, this project explores different compression techniques, including pruning,
quantization, and knowledge distillation, to reduce the size of deep neural networks while
preserving their accuracy. The evaluation is conducted on the CIFAR10 and SVHN datasets
using four pre-trained models. The findings will help identify the most suitable compression
technique for each dataset, enabling the deployment of efficient models on resource-limited
devices.



