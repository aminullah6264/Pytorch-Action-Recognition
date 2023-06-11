# Action Recognition in Video Sequences using Deep Bi-directional LSTM with CNN Features

https://ieeexplore.ieee.org/document/8121994/


# Features Extraction for YouTube Dataset
This Python script extracts features from videos in the YouTube Dataset using a pre-trained ResNet18 model. The extracted features can be used for training and testing LSTM models to perform action recognition.

Getting Started
Prerequisites
Python 3
PyTorch
scikit-learn
OpenCV
# Dataset
Download the YouTube Dataset from the following link: https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php

# Usage
featrues_extraction.py
Set the file path to your dataset in data_dir.
Define the number of frames to extract features from in num_frames.
Define the pre-processing steps for the images in transform.
Load the pre-trained ResNet18 model using models.resnet18(pretrained=True).cuda().
Remove the last layer of the ResNet18 model to obtain the feature extractor using torch.nn.Sequential(*list(resnet.children())[:-1]).
Loop over the videos in the dataset folder and extract features using the pre-trained ResNet18 model.
Split the samples into training and testing sets and convert the labels to numerical labels using a LabelEncoder.
Save the features and labels to numpy arrays.




LSTM Action Recognition Model
train_LSTM.py
This Python script implements a Long Short-Term Memory (LSTM) neural network for action recognition using features extracted from videos in the YouTube Dataset.

Load the features and labels from numpy arrays using torch.from_numpy(np.load('train_features.npy')).float() and torch.from_numpy(np.load('train_labels.npy')).
Define the LSTM model using LSTMClassifier or MultiLayerBiLSTMClassifier classes.
Define the loss function and optimizer using nn.CrossEntropyLoss() and optim.Adam() respectively.
Train the LSTM model using a for loop over the desired number of epochs and batches.
LSTM Model
LSTMClassifier: a simple LSTM classifier that takes as input a tensor of shape (batch_size, num_frames, input_size) and outputs a tensor of shape (batch_size, num_classes).
MultiLayerBiLSTMClassifier: a multi-layer bidirectional LSTM classifier that takes as input a tensor of shape (batch_size, num_frames, input_size) and outputs a tensor of shape (batch_size, num_classes).


## Please cite the following paper
@article{ullah2017action,
  title={Action recognition in video sequences using deep bi-directional LSTM with CNN features},
  author={Ullah, Amin and Ahmad, Jamil and Muhammad, Khan and Sajjad, Muhammad and Baik, Sung Wook},
  journal={IEEE access},
  volume={6},
  pages={1155--1166},
  year={2017},
  publisher={IEEE}
}



