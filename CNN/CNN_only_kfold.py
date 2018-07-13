# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:13:21 2018

@author: jinyang
"""

import tensorflow as tf
import numpy as np 
import cv2
import matplotlib.pyplot as plt

#hyper parameter definition 
LEARNING_RATE=0.0001
MAX_EPI = 50
BATCH_SIZE = 50
DROPOUT_PROB = 0.4
TRAIN_NUM = 4000
TEST_NUM = 1000
K_FOLD = 5
SUBSAMPLE_NUM = int (TRAIN_NUM / K_FOLD)

#input parameters
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
INPUT_CHANNEL = 3
FEATURE_NUM = 1904



def load_data(img_dir=None):
    image = []
    a = np.loadtxt(img_dir)
    temp = list(map(int,a[:,0:1]))
    for i in temp:
        image_read = cv2.imread('demo/'+str(i)+'.jpg')
        image.append(image_read)
    return (np.array(image))

def load_label(label_dir):
    label = np.loadtxt(label_dir)
    label = label[:,1:]
    return np.array(label)

def pretrement(input_data):
    with tf.variable_scope('pretreatment'):
        
        seven_by_seven = tf.layers.conv2d(input_data, filters=32, kernel_size = [7,7], strides = [2,2], padding = 'same',activation = tf.nn.relu,name = 'seven_by_seven')
        pooling_filter = tf.layers.max_pooling2d(seven_by_seven, pool_size = [3,3], strides = [1,1], padding = 'same',name = 'max_pool')
        
        # local_resp_norm 
        local_resp_norm1 = tf.nn.lrn(input = pooling_filter,name ='local_resp_norm1')
        
        one_by_one = tf.layers.conv2d(local_resp_norm1, filters = 32, kernel_size=[1,1], padding='same',activation=tf.nn.relu,name='one_by_one')
        three_by_three = tf.layers.conv2d(one_by_one, filters = 32, kernel_size=[3,3], padding='same',activation=tf.nn.relu,name='three_by_three')
        
        local_resp_norm2 = tf.nn.lrn(input=three_by_three, name='local_resp_norm2')

    return local_resp_norm2


def flatten_op(input_array):
    #suitable for only 1 unknown shape in input array
    #should change here if more than 1 dim is unknown
    shape=input_array.get_shape().as_list()
    dim=np.prod(shape[1:])
    flatten_array=tf.reshape(input_array,[-1,dim])
    return flatten_array

def Inception_layer(input_data):    
    #define the placeholders
    #input_data=tf.placeholder(tf.float32,[None,input_height,input_width,input_channel],name="input_placeholder")
    with tf.variable_scope('Inception_layer'):
        # 1x1
        one_by_one_1 = tf.layers.conv2d(input_data, filters=32, kernel_size=[1,1], padding='same',activation=tf.nn.relu,name='one_by_one_1')
        one_by_one_2 = tf.layers.conv2d(input_data, filters=32, kernel_size=[1,1], padding='same',activation=tf.nn.relu,name='one_by_one_2')
        one_by_one_3 = tf.layers.conv2d(input_data, filters=32, kernel_size=[1,1], padding='same',activation=tf.nn.relu,name='one_by_one_3')
        
        # 3x3
        three_by_three_1= tf.layers.conv2d(one_by_one_1, filters=32, kernel_size=[3,3], padding='same',activation=tf.nn.relu,name='three_by_three_1')
        three_by_three_2= tf.layers.conv2d(one_by_one_2, filters=32, kernel_size=[3,3], padding='same',activation=tf.nn.relu,name='three_by_three_2')
        three_by_three_3= tf.layers.conv2d(three_by_three_2, filters=32, kernel_size=[3,3], padding='same',activation=tf.nn.relu,name='three_by_three_3')

        # max pooling
        pooling_filter= tf.layers.max_pooling2d(input_data, pool_size=[3,3], strides=[1,1], padding='same',name='max_pool')
        pooling_conv= tf.layers.conv2d(pooling_filter,filters=32, kernel_size=[1,1], padding='same',activation=tf.nn.relu,name='conv_after_pooling')
        output_data = tf.concat([one_by_one_3, three_by_three_1, three_by_three_3, pooling_conv], axis=3)  # Concat in the 4th dim to stack
        #print (output_data.shape)

    return output_data

def model(input_data):

    with tf.variable_scope("pretreatment"):
        pre_t = pretrement(input_data)

    with tf.variable_scope('max_pool1'):
        max_pool_1=tf.layers.max_pooling2d(pre_t, pool_size=[3,3],strides=[2,2],padding='same',name='model_max_pool_1')

    with tf.variable_scope('inception_layer1'):
        incep_1=Inception_layer(max_pool_1)

    with tf.variable_scope('max_pool2'):
        max_pool_2=tf.layers.max_pooling2d(incep_1,pool_size=[3,3],strides=[2,2],padding='same',name='model_max_pool_2')

    with tf.variable_scope('inception_layer2'):
        incep_2=Inception_layer(max_pool_2)

    with tf.variable_scope('ave_pool'):
        ave_pool=tf.layers.average_pooling2d(incep_2,pool_size=[5,5],strides=[3,3],padding='same',name='model_ave_pool')
        total_input=flatten_op(ave_pool)
    #can add one 1*1 conv2d here
    
    with tf.variable_scope('fc1'):
        fc_layer_1=tf.layers.dense(total_input,units=512,activation=tf.nn.tanh,name='fc1')
        fc_layer_1=tf.nn.dropout(fc_layer_1,keep_prob=DROPOUT_PROB)

    #print (total_input.shape)
    with tf.variable_scope('fc2'):
        fc_layer_2=tf.layers.dense(fc_layer_1,units=2,name='fc2')
        
    return fc_layer_2

def main(argv=None):

    #prevent unknown bug
    tf.reset_default_graph()
    loss_gather=[]
    step_gather=[]
    
    #initialize the net
    
    #placeholders
    input_data=tf.placeholder(tf.float32,[None,INPUT_HEIGHT,INPUT_WIDTH,INPUT_CHANNEL],name='input_data')
    label=tf.placeholder(tf.float32,[None,2],name='label')
    pred = model(input_data) 
    #regularize on all variables
    var_all = tf.trainable_variables()
    l2_sum = 0
    for var_ in var_all:
        l2_sum += tf.nn.l2_loss(var_)
    
    loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = label)) + l2_sum
    gradient = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_func)

    #load image data and feature value
    #load label
    image_data = load_data('demo/label.txt')
    print("Image Load done!")
    label_input = load_label('demo/label.txt')
    print("Label Load done!")
    #uncomment for training
    ''' 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        
        #K-fold cross validation
        for kfold in range(K_FOLD):
            train_data = np.concatenate((image_data[0 : SUBSAMPLE_NUM * kfold],image_data[SUBSAMPLE_NUM * (kfold + 1):TRAIN_NUM]))
            validate_data = np.array(image_data[SUBSAMPLE_NUM * kfold : SUBSAMPLE_NUM * (kfold + 1)])
            label_train = np.concatenate([label_input[0 : SUBSAMPLE_NUM * kfold],label_input[SUBSAMPLE_NUM * (kfold + 1):TRAIN_NUM]])
            label_validate = np.array(label_input[SUBSAMPLE_NUM * kfold : SUBSAMPLE_NUM * (kfold + 1)])

            for i in range(MAX_EPI+1):
                step=0
                accuracy=[]
                while step < (TRAIN_NUM-SUBSAMPLE_NUM):
                    #print ("step:",step,"train_data_shape",train_data[step:step+BATCH_SIZE].shape)
                    feed_dict={input_data:train_data[step:step+BATCH_SIZE].reshape((BATCH_SIZE,INPUT_HEIGHT,INPUT_WIDTH,INPUT_CHANNEL)),
                              label:label_train[step:step+BATCH_SIZE].reshape((BATCH_SIZE,2))} 
                    _,loss_val=sess.run([gradient,loss_func],feed_dict=feed_dict)
                    step+=BATCH_SIZE
                
                print("K_FOLD:",kfold,"episode:",i," loss func value:",loss_val)
                
                if (i % 10 == 0):
                    saver.save(sess,'./modelWeights/newWeights_dropout/weights_At'+str(i)+'K_FOLD'+str(kfold+1)+'.ckpt')
                loss_gather.append(loss_val)
                step_gather.append(i)
            
            #plot the loss function
            plt.plot(step_gather,loss_gather)
            plt.xlabel('step')
            plt.ylabel('loss_func')
            
            result=[]
            for t in range(SUBSAMPLE_NUM):
                feed_dict={input_data:validate_data[t].reshape((1,INPUT_HEIGHT,INPUT_WIDTH,INPUT_CHANNEL))}
                classification = sess.run(pred, feed_dict)
                result.extend(classification.argmax(axis=1))
            label_pos=np.argmax(label_validate,axis=1) 
            boolResult = np.array(result)==np.array(label_pos)[0:4000]
            test_accu = len(boolResult[boolResult==True])/TEST_NUM
            print (test_accu)
            accuracy.append(test_accu)
    '''


#uncomment for testing
    saver=tf.train.Saver()
    saved_episode=[10,20,30,40,50]
    accuracy=[]
    with tf.Session() as sess:
        for i in saved_episode:
            result=[]
            saver.restore(sess, "./modelWeights/newWeights_dropout/weights_At"+str(i)+"K_FOLD5"+".ckpt")
            #writer = tf.summary.FileWriter("logs/", sess.graph)
            for i in range(4000,5000):
                feed_dict={input_data:image_data[i].reshape((1,INPUT_HEIGHT,INPUT_WIDTH,INPUT_CHANNEL))}
                classification = sess.run(pred, feed_dict)
                #print(i)
                #print(classification.argmax(axis=1))
                result.extend(classification.argmax(axis=1))
            label_pos=np.argmax(label_input,axis=1) 
            boolResult = np.array(result)==np.array(label_pos)[4000:5000]
            print (len(boolResult[boolResult==True]))
            accuracy.append(len(boolResult[boolResult==True])/1000)
    plt.plot(saved_episode,accuracy)
    plt.xlabel("episode")
    plt.ylabel("train accuracy")
   
if __name__ == "__main__":
    main()

