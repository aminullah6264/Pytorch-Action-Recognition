# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:55:09 2017

@author: AMIN
"""

import tensorflow as tf
from tensorflow.contrib import rnn

from numpy import genfromtxt
import numpy as np

import os, fnmatch
import sklearn.model_selection as sk
import time

n_classes = 51
chunk_size =1000
n_chunks =6
rnn_size = 128
 

n_nodes_hl1 = 256
n_nodes_hl2 = 128
n_nodes_hl3 = 64




x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')


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
    weights1 = tf.Variable(tf.random_normal([2*rnn_size, n_classes]),name="weights1")
    
    biases1 =  tf.Variable(tf.random_normal([n_classes]),name="biases1")
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights1) + biases1

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
    
    
#
#    x = tf.transpose(x, [1,0,2])
#    x = tf.reshape(x, [-1,chunk_size])
#    x = tf.split (x,n_chunks, 0)
#    lstm_cell = rnn.BasicLSTMCell(rnn_size)
#    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#    
#    
#
#    weights1=tf.Variable(-1.0, validate_shape=False,name="weights1")   #,mean=0.2, stddev=0.2
#    biases1=tf.Variable(-1.0, validate_shape=False,name="biases1")
#    l1 = tf.add(tf.matmul(outputs[-1],weights1), biases1)
#    l1=tf.sigmoid(l1)
#
#
#    weights2=tf.Variable(-1.0, validate_shape=False,name="weights2")
#    biases2=tf.Variable(-1.0, validate_shape=False, name="biases2")
#    l2 = tf.add(tf.matmul(l1,weights2), biases2)
#    l2=tf.sigmoid(l2)
#         
#    weights3=tf.Variable(-1.0, validate_shape=False,name="weights3")
#    biases3=tf.Variable(-1.0, validate_shape=False, name="biases3")
#    l3 = tf.add(tf.matmul(l2,weights3), biases3)
#    l3=tf.sigmoid(l3)
#    
#    
#    weightsOutput=tf.Variable(-1.0, validate_shape=False,name="weightsOutput")
#    biasesOutput=tf.Variable(-1.0, validate_shape=False,name="biasesOutput")
#    output = tf.matmul(l3,weightsOutput)+ biasesOutput
#   # output=tf.sigmoid(output)
#
#             
#    return output

def train_recurrnet_neural_network(x):

    prediction= recurrent_neural_network(x)
    tf.device('/gpu:0')
    ConfussionMatrix=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "/home/imlab/Desktop/RNN Codes/Action BiDirectional Model For HMD Dataset/model.chk")
        #print(sess.run(tf.all_variables()))
        
        for mm in range(1,2,1):
            hist1=np.zeros(51)
            TotalResult=np.zeros(51)
            clNew='*class_'+str(mm)+'.csv'
            ClassFiles=fnmatch.filter(os.listdir('/home/imlab/Desktop/RNN Codes/Action DataSets/HMD Dataset/Test/'), clNew)
            print('class Number=',mm)
            for t in range(0,len(ClassFiles),1):
                time_start = time.clock()
                X_test = genfromtxt('/home/imlab/Desktop/RNN Codes/Action DataSets/HMD Dataset/Test/'+str(ClassFiles[t]), delimiter=',')
                lenghtX,dd=X_test.shape
                time_start = time.clock()
                labled=sess.run(tf.argmax(prediction,1), feed_dict={x: X_test.reshape((-1,n_chunks, chunk_size))})
                
                hist1[:]=0
                for k in range(len(labled)):
                    hist1[labled[k]]  = hist1[labled[k]] +1
                time_elapsed = (time.clock() - time_start)
                print(time_elapsed)
    #            for k in range(len(hist1)):
    #                print(k,' Labled =',hist1[k])
    #            for k in range(len(hist1)):
    #                print(k,' Labled =',hist1[k]/lenghtX)
    #                if hist1[k]/lenghtX >= 0.5:
                TotalResult[np.argmax(hist1)]=TotalResult[np.argmax(hist1)]+1
    #
            ConfussionMatrix.append(TotalResult)   
#            for k in range(len(TotalResult)):
#                print(k," =" ,TotalResult[k])   
    return ConfussionMatrix

ConfussionMatrix=train_recurrnet_neural_network(x)



        # OLD:
#        sess.run(tf.initialize_all_variables())
        # NEW:
#    x = tf.Variable(-1.0, validate_shape=False, name="weights")
#    y = tf.Variable(-1.0, validate_shape=False, name="biases")
#    with tf.Session() as session:
#        session.run(tf.global_variables_initializer())
#        saver = tf.train.Saver()
#        saver.restore(session, "/home/imlab/Desktop/NewFolder/Age With New Features With 256D/model.chk")
#        print(session.run(tf.all_variables()))
#        sess.run(tf.global_variables_initializer())
##        saver=tf.train.Saver()
#        saver = tf.train.import_meta_graph('/home/imlab/Desktop/NewFolder/Age With New Features With 256D/model.ckpt.meta')
#        
##        print('weights =', sess.run(weights))
##        print('biases =', sess.run(biases))
#        print("Loading Parameter from  checkpoint_file ......" ) 
#        saver.restore(sess,tf.train.latest_checkpoint('/home/imlab/Desktop/NewFolder/Age With New Features With 256D/'))
##        saver.restore(sess, tf.train.latest_checkpoint('/home/imlab/Desktop/NewFolder/Age With New Features With 256D/')
#        print('weights =', sess.run(weights))
#        print('biases =', sess.run(biases))
##        sess = tf.Session()
#new_saver = tf.train.import_meta_graph('my-model.meta')
#new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#        all_vars = tf.get_collection('vars')
#        print(all_vars)
#        for v in all_vars:
#            v_ = sess.run(v)
#            print(v_)
#        print('weights =', sess.run(weights))
#        print('biases =', sess.run(biases))
        
        # Create some variables.
 

#import os, fnmatch
#fnmatch.filter(os.listdir('101 Test Files/'), '*class_11.csv')
# 
# Add ops to save and restore all the variables.
#    saver = tf.train.Saver()
#
## Later, launch the model, use the saver to restore variables from disk, and
## do some work with the model.
#    with tf.Session() as sess:
#  # Restore variables from disk.
#      saver.restore(sess, "/home/imlab/Desktop/NewFolder/Age With New Features With 256D/model.ckpt")
#      print("Model restored.")
  # Do some work with the model