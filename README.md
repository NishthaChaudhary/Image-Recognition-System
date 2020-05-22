# Image-Recognition-System
Image Recognition System creates a deep learning model that learns from a image dataset-cifar10; This will classify a new image based on the learnings/knowledge of the model. In the second part, features are extracted from a pre-trained model. Based on the features a new model is created that learns and classify the image.

## Dataset:

The dataset includes thousands of pictures of 10 different kinds of objects. Each image in the dataset includes a matching label so we know what kind of image it is. The images of the dataset are 32 pixels by 32 pixels.

## Coding a Neaural Network with Keras

### 1. Add Dense layers:

A model network is built with the densely connected layers with two input nodes, then a layer of three nodes, a second layer of three nodes and finally an output layer with one node. A sequential model is created. We have 50000 training images with 10000 validation data images.
#### model=Sequential()

![image](https://user-images.githubusercontent.com/54689111/82656129-3bbeb900-9bf1-11ea-9ebe-932dfd04e8ca.png)


### 2. Add Convolutional Layers:

To improve the neural network so that it can recognize the objects in any position in the image, we will add conovulutional layers to manage transational invariance. Window of 3*3 is passed to pass it over the image. We will create several comvolutional layers so that they can look for different patterns.

### 3. Add MAx-Pooling:

This process will down sample the data of array created from the previous step. This grid is first divided into two-by-two squares, then find the largest number, and eventually create a new array that saves the largest number from that grid. This downsamples the data. 

### 4. Add Dropout:

This means throwing away the data randomly to make the learning of the model more robust and efficient. I have used 25% dropout value. Convolutional layers+ Max-Pooling layer+ Dropout makes a convulutional bloc. I have added two such blocks.

## Feature Extraction from a Pre-trained model- vgg16

### Sample Image:

![image](https://user-images.githubusercontent.com/54689111/82658218-a0c7de00-9bf4-11ea-8a6d-16f5ba697046.png)

### Output:

![image](https://user-images.githubusercontent.com/54689111/82658290-bd641600-9bf4-11ea-9b1b-9e40389c0162.png)

### 1. Use Transfer Learning

With transfer learning, we are going to start with NN that'a already been trained to recognize objects from a large dataset like ImageNet. We keep all tha convolutional layers that detect the patterns  of the data and chop off the last layer that maps the image with its category or label.

### 2. Extract features with pre-trained model:

I have two folders: dog folder that conatins dog images with 64 by 64 pixel from the image net dataset. Next folder is not_dog folder that images other than dogs. Now, we use the pretrained model to extract features from our training images. We will create another array for labels. 

![image](https://user-images.githubusercontent.com/54689111/82657234-0915c000-9bf3-11ea-9c53-aecfcbb7c8e9.png)

### 3. Training a new neaural network with extracted features

After applying convolutional layers i.e extracting features from the pretrained model, we will apply the Dense layer to determine the class of the image.

![image](https://user-images.githubusercontent.com/54689111/82657388-4712e400-9bf3-11ea-8584-ed48458e1ed7.png)


