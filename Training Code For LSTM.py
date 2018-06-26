# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:50:43 2017

@author: AMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:13:12 2017

@author: AMIN
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sklearn.model_selection as sk

#X_train, X_Validation, Y_train, Y_Validation = sk.train_test_split(Features,lables,test_size=0.33,random_state = 42) #, random_state = 1
X_train=TrainData
Y_train=Tlables
X_Validation=ValData
Y_Validation=Vlables


hm_epochs = 1000
n_classes = 6
batch_size = 128
batch_size_val=1024
chunk_size =676
n_chunks =6
rnn_size = 128
 

#n_nodes_hl1 = 256
#n_nodes_hl2 = 128
#n_nodes_hl3 = 64


trainSamples,FeaturesLength=Y_train.shape
ValidationSamples,FeaturesLength=Y_Validation.shape
loss=[];
Val_Accuracy=[];   

with tf.name_scope('Inputs'):
    x = tf.placeholder('float', [None, None,chunk_size],name="Features")
    y = tf.placeholder('float',name="Lables")

def recurrent_neural_network(x):
    
    
#  
#    W = {
#            'hidden': tf.Variable(tf.random_normal([chunk_size, rnn_size])),
#            'output': tf.Variable(tf.random_normal([rnn_size, n_classes]))
#        }
#    biases = {
#            'hidden': tf.Variable(tf.random_normal([rnn_size], mean=1.0)),
#            'output': tf.Variable(tf.random_normal([n_classes]))
#        }
#
#
#    x = tf.transpose(x, [1,0,2])
#    x = tf.reshape(x, [-1,chunk_size])
#    x = tf.nn.relu(tf.matmul(x, W['hidden']) + biases['hidden'])
#    x = tf.split (x,n_chunks, 0)
#    # new shape: n_steps * (batch_size, n_hidden)
#
#    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
#    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
#    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
#    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
#    # Get LSTM cell output
#    outputs, final_states = tf.contrib.rnn.static_rnn(lstm_cells, x, dtype=tf.float32)
#    # Get last time step's output feature for a "many to one" style classifier,
#    # as in the image describing RNNs at the top of this page
##    lstm_last_output=tf.transpose(outputs, [1,0,2])
#    # Linear activation
#    return tf.matmul(outputs[-1], W['output']) + biases['output']
    
#####################################################################  
    
    
    
  
   
      # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_chunks, 1)
    
    
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    
    
    lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_4 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_3, lstm_cell_4], state_is_tuple=True)
    
    
    # Define lstm cells with tensorflow
    # Forward direction cell
#    lstm_fw_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)
#    # Backward direction cell
#    lstm_bw_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)
        
        
   # Hidden layer weights => 2*n_hidden because of forward + backward cells
    weights = tf.Variable(tf.random_normal([2*rnn_size, n_classes]),name="weights1")
    
    biases =  tf.Variable(tf.random_normal([n_classes]),name="biases1")

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases
#    

######################################################################
    
#    weights = {
#    # Hidden layer weights => 2*n_hidden because of forward + backward cells
#    'out': tf.Variable(tf.random_normal([2*rnn_size, n_classes]))
#    }
#    biases = {
#        'out': tf.Variable(tf.random_normal([n_classes]))
#    }
#      # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#    x = tf.unstack(x, n_chunks, 1)
#
#    # Define lstm cells with tensorflow
#    # Forward direction cell
#    lstm_fw_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)
#    # Backward direction cell
#    lstm_bw_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)
#
#    # Get lstm cell output
#    try:
#        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                              dtype=tf.float32)
#    except Exception: # Old TensorFlow version only returns outputs not states
#        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                        dtype=tf.float32)
#
#    # Linear activation, using rnn inner loop last output
#    return tf.matmul(outputs[-1], weights['out']) + biases['out']
#    
#    
#    
    
    
 #########################################################################   
    
    
#    x = tf.transpose(x, [1,0,2])
#    x = tf.reshape(x, [-1,chunk_size])
#    x = tf.split (x,n_chunks, 0)
#    lstm_cell = rnn.BasicLSTMCell(rnn_size)
#    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#    
#    
#
#    weights1=tf.Variable(tf.random_normal([rnn_size, n_nodes_hl1], stddev=0.2),name="weights1")   #,mean=0.2, stddev=0.2
#    biases1=tf.Variable(tf.random_normal([n_nodes_hl1], stddev=0.2), name="biases1")
#    l1 = tf.add(tf.matmul(outputs[-1],weights1), biases1)
#    l1=tf.sigmoid(l1)
#
#
#    weights2=tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.2),name="weights2")
#    biases2=tf.Variable(tf.random_normal([n_nodes_hl2], stddev=0.2), name="biases2")
#    l2 = tf.add(tf.matmul(l1,weights2), biases2)
#    l2=tf.sigmoid(l2)
#         
#    weights3=tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.2),name="weights3")
#    biases3=tf.Variable(tf.random_normal([n_nodes_hl3], stddev=0.2), name="biases3")
#    l3 = tf.add(tf.matmul(l2,weights3), biases3)
#    l3=tf.sigmoid(l3)
#    
#    
#    weightsOutput=tf.Variable(tf.random_normal([n_nodes_hl3, n_classes], stddev=0.2),name="weightsOutput")
#    biasesOutput=tf.Variable(tf.random_normal([n_classes], stddev=0.2), name="biasesOutput")
#    output = tf.matmul(l3,weightsOutput)+ biasesOutput
#   # output=tf.sigmoid(output)

             
#    return output

def train_recurrnet_neural_network(x):
    

    
    
    prediction= recurrent_neural_network(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    best_accuracy = 0.0
   
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits
                      (logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        tf.device('/gpu:0')
        sess.run(tf.global_variables_initializer())
        
       
#        print(sess.run(weights))
                          
        kk=0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            valdd=[]  
            k=0;
            for _ in range(int(trainSamples/batch_size)):
                epoch_x = X_train[k:k+batch_size,:]
                epoch_y = Y_train[k:k+batch_size,:]
                epoch_x= epoch_x.reshape((batch_size, n_chunks, chunk_size ))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                k=k+batch_size
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            loss.append(epoch_loss)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            
            accuracy_out = (accuracy.eval({x:X_Validation.reshape((-1,n_chunks, chunk_size)), y:Y_Validation}))


                    
                    
#            kk=0
#            for _ in range(int(ValidationSamples/batch_size_val)):
#                valdd.append(accuracy.eval({x:X_Validation[kk:kk+batch_size_val,:].reshape((-1,n_chunks, chunk_size)), y:Y_Validation[kk:kk+batch_size_val,:]}))
#                kk = kk+batch_size_val
#                if kk > ValidationSamples:
#                    kk=0
#            accuracy_out=np.mean(valdd)
            Val_Accuracy.append(accuracy_out)
            print('Validation Accuracy : ',accuracy_out,'  ||| Best Accuracy :',best_accuracy)
#            if  accuracy_out > best_accuracy:
#                    best_accuracy=accuracy_out
#                    saver = tf.train.Saver() 
#                    save_path = saver.save(sess, "D:\\New Experiments\\New Action Recognition\\Checkpoints\\model.chk")
#                    print("Model saved in file: %s" % save_path)
                    
         
        
        #Save the variables to disk.
#        save_path = saver.save(sess, "D:\\Speech Project\\Dataset\\BerlinImages\\BerlinImages\\1_Singleimages\\RNN Model For 257x45 double data spects\\model.ckpt")
#        print("Best Accuracy ==  " ,best_accuracy)
       # merged = tf.summary.merge_all()
       # writer=tf.summary.FileWriter("C:\\Users\\AMIN\\Anaconda2\\envs\\py35\\Lib\\site-packages\\tensorflow\\tensorboard\\otherLogs",sess.graph)
        

train_recurrnet_neural_network(x)



'''

import tables

file = tables.open_file('D:\\Action and Scenes\\Code\\DataHMDDATASET.mat')

lon = file.root.TotalFeatures[:]

import numpy

TotalFeaturesX=numpy.transpose(lon)
'''

''''
import h5py

import numpy as np
filepath = '/home/imlab/Desktop/RNN Codes/6SeqDataWith5FrameJump101dataSet.mat'
arrays = {}
f = h5py.File(filepath)
i=1
for k, v in f.items():
    if(i==1):
        Features=np.array(v)
    if(i==2):
        lables=np.array(v)
    i=i+1
    print(v)
    
Features=np.transpose(Features)
lables=np.transpose(lables)

'''
''''
import h5py

import numpy as np
filepath = '/home/imlab/Desktop/RNN Codes/Action DataSets/101 Dataset/ErrorFreeTestTrain101DataSet.mat'
arrays = {}
f = h5py.File(filepath)
i=1
for k, v in f.items():
    if(i==1):
        Tlables=np.array(v)
    if(i==2):
        TrainData=np.array(v)
    if(i==3):
        ValData=np.array(v)
    if(i==4):
        Vlables=np.array(v)
    i=i+1
    print(v)
    
TrainData=np.transpose(TrainData)
ValData=np.transpose(ValData)
Tlables=np.transpose(Tlables)
Vlables=np.transpose(Vlables)

'''



############################################################################




''''
import h5py

import numpy as np
filepath = '/home/imlab/Desktop/RNN Codes/Action DataSets/101 Dataset/101TrainData.mat'
arrays = {}
f = h5py.File(filepath)
i=1
for k, v in f.items():
    if(i==1):
        TrainData1=np.array(v)
    if(i==2):
        TrainData2=np.array(v)
    if(i==3):
        TrainData3=np.array(v)
    if(i==4):
        TrainData4=np.array(v)
    if(i==5):
        TrainData5=np.array(v)
    if(i==6):
        TrainData6=np.array(v)
    if(i==7):
        TrainData7=np.array(v)
    if(i==8):
        TrainData8=np.array(v)
    if(i==9):
        TrainData9=np.array(v)
    if(i==10):
        TrainData10=np.array(v)
    i=i+1
    print(v)
    
TrainData1=np.transpose(TrainData1)
TrainData2=np.transpose(TrainData2)
TrainData3=np.transpose(TrainData3)
TrainData4=np.transpose(TrainData4)
TrainData5=np.transpose(TrainData5)
TrainData6=np.transpose(TrainData6)
TrainData7=np.transpose(TrainData7)
TrainData8=np.transpose(TrainData8)
TrainData9=np.transpose(TrainData9)
TrainData10=np.transpose(TrainData10)

TrainData=np.concatenate([TrainData1,TrainData3,TrainData4,TrainData5,TrainData6,TrainData7,TrainData8,TrainData9,TrainData10,TrainData2]);


'''


''''
import h5py

import numpy as np
filepath = '/home/imlab/Desktop/RNN Codes/Action DataSets/101 Dataset/101ValData.mat'
arrays = {}
f = h5py.File(filepath)
i=1
for k, v in f.items():
    if(i==1):
        ValData=np.array(v)
    i=i+1
    print(v)
    
ValData=np.transpose(ValData)

'''