# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 08:54:35 2017

@author: zhenm
"""
import tensorflow as tf
import numpy as np
def help():
    """
    ********Python 3.5 have tensorflow module************
    hi, welcome to use simple CNN, this module is very easy to use compared with Tensorflow.
    Some students who know little about Tensorflow or don't have a thick Python background will feel be confused by tensorflow's complex grammer.
    So maybe they cannot get the real meaning of CNN and cannot use it quickly.
    As a result, I made this module for them to focus their attention on CNN not Tensorflow.
    This is a very simple module. Remember these things.
    First, don't not use functin which begin with '_'
    Second, the all captical lettle can be revies.
    Third, 'x,y_,y,keep_prob' is key words, so don't use them as your variables' name
    Forth, 
    initialize_cnn(train_x,train_y)
    reshape_image(x,image_width,image_height)
    nn_layer(input_tensor, weight_size, layer_name, conv='conv2d', conv2d_strides=[1,1,1,1],act='relu',
             pool='pool',pool_ksize=[1,2,2,1],pool_strides=[1,2,2,1],keep_prob=1.0)
    train_cnn(x,y_,keep_prob,y,train_x,train_y,LEARNING_RATE=1e-4,TRAINING_ITERATIONS = 10000,BATCH_SIZE=100,KEEP_PROB=0.25)
    test_cnn(x,keep_prob,y,test_x,num_label,BATCH_SIZE=100)
    So, you can use tensorflow in a very easy way and here is an example
    
    #This part is to process raw data. The data is mnist, a very wide used data set.
    import simple_CNN as sc
    import pandas as pd
    import numpy as np
    data = pd.read_csv('G:\\digit\\train.csv')
    images = data.iloc[:,1:].values
    images = images.astype(np.float)
    images = np.multiply(images, 1.0 / 255.0)
    train_x=images
    train_y=data.iloc[:,0].values.ravel()
    test_images = pd.read_csv('G:\\digit\\test.csv').values
    test_images = test_images.astype(np.float)
    test_x = np.multiply(test_images, 1.0 / 255.0)
    
    #This part is CNN, we have a 4 layers CNN.
    x,y_,keep_prob=sc.initialize_cnn(train_x,train_y)
    image=sc.reshape_image(x,28,28)
    h_1=sc.nn_layer(image,[5,5,1,32],'layer_conv_1')
    h_2=sc.nn_layer(h_1,[5,5,32,64],'layer_conv_2')
    h_3=sc.nn_layer(h_2,[7*7*64,1024],'full_connection',conv='matmul',act='relu',pool='dropout',keep_prob=keep_prob)
    h_4=sc.nn_layer(h_3,[1024,10],'output',conv='matmul',act='softmax',pool=False)
    y=h_4
    sc.train_cnn(x,y_,keep_prob,y,train_x,train_y,TRAINING_ITERATIONS=1000)
    predicted_proba=sc.test_cnn(x,keep_prob,y,test_x,10)
    predicted_lables = np.argmax(predicted_proba,axis=1)
    #Now we have our answer
    #Enjoy you CNN model
    #If you have any question, please contact me.
    """
def _variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def initialize_cnn(train_x,train_y):
    # input & output of NN
    x = tf.placeholder('float', shape=[None,train_x.shape[1]])
    # labels
    y_ = tf.placeholder('float', shape=[None,np.unique(train_y).shape[0]])
    #drop_prob
    keep_prob=tf.placeholder('float')
    return x,y_,keep_prob

def reshape_image(x,image_width,image_height):
    image=tf.reshape(x,[-1,image_width,image_height,1])
    return image
# weight initialization
def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, weight_size, layer_name, conv='conv2d', conv2d_strides=[1,1,1,1],act='relu',
             pool='pool',pool_ksize=[1,2,2,1],pool_strides=[1,2,2,1],keep_prob=1.0):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  if pool=='dropout':
      input_tensor=tf.reshape(input_tensor,[-1,*weight_size[0:-1]])
  #input_tensor=tf.reshape(input_tensor,[-1,*weight_size[0:-1]])
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        weights = _weight_variable(weight_size)
        _variable_summaries(weights)
    with tf.name_scope('biases'):
        biases = _bias_variable([weight_size[-1]])
        _variable_summaries(biases)
    with tf.name_scope('conv_Wx_plus_b'):
        if conv=='matmul':
            h_conv= tf.matmul(input_tensor,weights) + biases
        elif conv=='conv2d':
            h_conv = tf.nn.conv2d(input_tensor,weights,strides=conv2d_strides,padding='SAME') + biases
        tf.summary.histogram('h_conv',h_conv)
    with tf.name_scope('activations'):
        if act=='relu':
            h_act=tf.nn.relu(h_conv,name='relu')
        elif act=='softmax':
            h_act=tf.nn.softmax(h_conv)
        tf.summary.histogram('activations', h_act)
    with tf.name_scope('pool'):
        if pool=='pool':
            h_pool=tf.nn.max_pool(h_act,ksize=pool_ksize,strides=pool_strides,padding='SAME')
            tf.summary.histogram('h_pool', h_pool)
        elif pool=='dropout':
            h_pool=tf.nn.dropout(h_act,keep_prob)
            tf.summary.histogram('h_dropout',h_pool)
        else:
            h_pool=h_act
    return h_pool

# convert class labels from scalars to one-hot vectors
def _dense_to_one_hot(train_y):
    num_labels = train_y.shape[0]
    num_classes=np.unique(train_y).shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + train_y.ravel()] = 1
    return labels_one_hot

def train_cnn(x,y_,keep_prob,y,train_x,train_y,LEARNING_RATE=1e-4,TRAINING_ITERATIONS = 10000,BATCH_SIZE=100,KEEP_PROB=0.25):
    train_y=_dense_to_one_hot(train_y)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_x.shape[0]
    # serve data by batches
    def next_batch(batch_size):
        nonlocal train_x
        nonlocal train_y
        nonlocal index_in_epoch
        nonlocal epochs_completed
        start = index_in_epoch
        index_in_epoch += batch_size 
        # when all trainig data have been already used, it is reorder randomly    
        if index_in_epoch > num_examples:
            # finished epoch
            epochs_completed += 1
            # shuffle the data
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            train_x = train_x[perm]
            train_y = train_y[perm]
            # start next epoch
            start = 0
            index_in_epoch = batch_size
            assert batch_size <= num_examples
        end = index_in_epoch
        return train_x[start:end],train_y[start:end]

    # start TensorFlow session
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()

    sess.run(init)
    merged = tf.summary.merge_all()
    # visualisation variables
    train_accuracies = []
    x_range = []
    
    display_step=1
    
    for i in range(TRAINING_ITERATIONS):
    
        #get new batch
        batch_xs, batch_ys = next_batch(BATCH_SIZE)        
    
        # check progress on every 1st,2nd,...,10th,20th,...,100th... step
        if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
            
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                      y_: batch_ys, 
                                                      keep_prob: 1.0})       
            print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
            train_accuracies.append(train_accuracy)
            x_range.append(i)
            
            # increase display_step
            if i%(display_step*10) == 0 and i:
                display_step *= 10
        # train on batch
        sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: KEEP_PROB})

def test_cnn(x,keep_prob,y,test_x,num_label,BATCH_SIZE=100):
    predicted_proba = np.zeros((test_x.shape[0],num_label))
    for i in range(0,test_x.shape[0]//BATCH_SIZE):
        predicted_proba[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = y.eval(feed_dict={x: test_x[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                    keep_prob: 1.0})
    return predicted_proba