# BidirectionalLSTM
Action Recognition in Video Sequences using Deep Bi-directional LSTM with CNN Features

https://ieeexplore.ieee.org/document/8121994/


We have used caffe mode to extract deep features from video using matlab script "oneFileFeatures..."
Each CSV file represents features of one video.
which is split using "TrianTestSpit.m"
each CSV in train data is combined to create one CSV file for each class using "EachClassCSV"
The train and validation split is done on Train Data with "EachClassCSV" files and it also give us Labels. which convert to one hot using "oneHotLabeling"
and finally, we use the "Training Code for LSTM.py" this code contains simple LSTM, Multi-layer LSTM, and Multi-layer Bidirectional LSTM.


Please cite the folloing papers

Ullah, A., Ahmad, J., Muhammad, K., Sajjad, M., & Baik, S. W. (2018). Action Recognition in Video Sequences using Deep Bi-Directional LSTM With CNN Features. IEEE Access, 6, 1155-1166.
