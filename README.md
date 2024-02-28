# Airbus Ship Detection
This project addresses the Airbus Ship Detection Challenge on Kaggle, focusing on developing a machine learning model to detect ships in satellite images. 

## Introduction

This project aims to create a U-Net based model designed to precisely delineate ships within satellite imagery. Utilizing genuine Airbus images paired with their corresponding masks, each having dimensions of 768x768 pixels, we focus on the segmentation task where the masks employ a simple binary color scheme: white pixels represent ships, and black pixels depict the surrounding environment.

## Prerequisites

❯ Python 3.8

❯ Git

❯ requirements.txt


## Installation

1. To install the project, you need to clone the repository first:
```
$ mkdir ~/workdir
$ cd ~/workdir
$ git clone https://github.com/CaCuCkA/airbus-ship-detection
$ cd traffic-server
```

2. Setup python virtual environment and install its requirements:
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```

3. Download Airbus dataset from [Kaggle website](https://www.kaggle.com/c/airbus-ship-detection/data) and make new `data` directory for them. 
> **Note** <br>
> If yoy have `kaggle` on your laptop, you can use this command:
> ```bash
> $ kaggle competitions download -c airbus-ship-detection
> ```

The project should follow such structure:

```
airbus-ship-detection/
├── data/
│ ├── test_v2/
│ ├── train_v2/
│ ├── sample_submission_v2.csv
│ └── train_ship_segmentations_v2.csv
├── misc/
│ ├── __init__.py
│ ├── constants.py
│ ├── data_processing.py
│ ├── mask_rle_processing.py
│ ├── metrics.py
│ ├── segment_loader.py
│ └── unet.py
├── airbus_ship_detection_eda.ipynb
├── evaluate.py
├── LICENSE
├── README.md
├── requirements.txt
└── train.py

```

## Usage

For easy testing the project perforamnce I split model training and evaluating in `train.py` and `evaluate.py`. This files contain only high-level code if you want to see how I was implementing any of specific function you should research `misc` folder. 

1. To train model you should this such code:
```
$ python3 train.py
```

2. To evaluate model on the test set:
```
$ python3 evaluate.py
```

## Architecture
![image](https://www.frontiersin.org/files/Articles/841297/fnagi-14-841297-HTML-r2/image_m/fnagi-14-841297-g001.jpg)

The ship detection model employs the U-Net architecture, renowned for its image segmentation capabilities. U-Net effectively balances local and global context, incorporating an encoder for feature extraction and a decoder for upsampling, culminating in precise segmentation maps.

Here's an expanded breakdown that articulates the functionality of each component within the U-Net architecture:

#### 1. Input Layer
Acts as the entry point for the images to be processed by subsequent layers of the model

#### 2. Encoder Layer
The encoder pathway is responsible for feature extraction, comprising several key components:
* __Convolutional Layers:__ Apply filter (kernel) to the input image to extract various features, such as _edges_, _textures_, and other _relevant_ _patterns_. This layer consists of 64 filters, followed by ReLU activation and padding.

* __Activation functio (ReLU):__ ntroduces non-linearity into the model, which is crucial because real-world data is inherently non-linear. By applying ReLU, a model can learn complex patterns and relationships within the data. 

* __Pooling Layer:__ Reduce the spatial dimensions (width and height) of the feature maps while retaining the most critical information. U-Net usually uses __Max Polling__ of size (2, 2). 

#### 3. Bottleneck
It is the deepest point of the network. It located between encoder and decoder and  process the most abstract representations of the input data, serving as a bridge that condenses and then expands the feature information.


#### 4. Decoder Pathway
The decoder pathway reconstructs the segmentation map from the encoded features, including:

* __Unsampling Layer:__ Increase the spatial dimensions of the feature maps to reconstruct the size of the original image.

* __Concatenation with Skip Connections:__ Combines features from the encoder pathway with the upsampled features.

* __Convolutional Layer:__ Further refine the features and reconstruct the detailed segmentation map.

* __Activation Function (ReLU)__

#### 5. Output Layer
 Produces the final segmentation map, indicating the precise locations and boundaries of ships within the images.

 ## Training
The training process for our ship detection model combines binary cross-entropy loss with the Dice coefficient to fine-tune model parameters, aiming for precise pixel classification and accurate ship segmentation. I employ the Adam optimizer with its default settings, benefiting from its adaptive learning rate to efficiently navigate the optimization landscape. 
