#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:42:07 2018

@author: a1anzvvy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Directory
data_dir = ''   # 样本数据存储的路径
log_dir = ''    # 输出日志保存的路径

#Import Data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Define Model
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#y = tf.matmul(x,W)+b
y = tf.nn.softmax(tf.matmul(x,W)+b)

# Define Loss and Optimizer
y_ = tf.placeholder(tf.float32,[None,10])

# 
#cross_entropy = -tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialize Model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Train
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys})
    if(i%1000 == 0):
        print(i)


correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print (sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


sess.close()