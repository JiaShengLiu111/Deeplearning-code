# _*_ coding:utf-8 _*_
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题

import os
sys.path.append(os.path.dirname(os.getcwd()))  # add the upper level directory into the pwd
import tf_fun
import tensorflow as tf
import tensorflow.contrib.slim as slim

import tf_fun

class AkinVgg:
    """
    the network similar to Vgg
    """
    def __init__(self,class_num,width_multiplier=1,dropout_rate=0.2):
        """
        parameters:
           inputs:the input of the network
           is_training:trainable or not
           class_num:the final class number
           dropout_rate:dropout rate
        """ 
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        self.width_multiplier = width_multiplier
        
        # construct placeholder
        self.inputs = tf.placeholder(tf.float32, [None, 112, 112, 3])  # the input of the network
        self.labels = tf.placeholder(tf.float32, [None, self.class_num])  # the labels of train sampels
        self.is_training = tf.placeholder(tf.bool)  # trainable or not
        
        # build the network
        self.prob = self.build(self.inputs)
        
        # construct loss Function
        self.cost = -tf.reduce_mean(self.labels*tf.log(tf.clip_by_value(self.prob,1e-10,1.0)))  

    def build(self,inputs,scope="MobileNetV1"): 
        assert inputs.get_shape().as_list()[1:]==[112,112,3], 'the size of inputs is incorrect!'
        
        # start to build the model
        net = inputs 
        print ("start-shape:"+str(net.shape)) 
        
        # 第一区
        tf_op = tf_fun.tf_fun(self.is_training)
        net = tf_op.conv_layer(net, kernel_num=32, kernel_size=3, stride=1, layer_name="conv_1",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_1/bn")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=32, kernel_size=3, stride=1, layer_name="conv_2",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_2/bn")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool1")
        print ("the 1th stage-shape："+str(net.shape))
        
        # 第二区
        net = tf_op.conv_layer(net, kernel_num=32, kernel_size=3, stride=1, layer_name="conv_3",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_3/bn")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=32, kernel_size=3, stride=1, layer_name="conv_4",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_4/bn")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool2") 
        print ("the 2th stage-shape："+str(net.shape))
        
        # 第三区
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv_5",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_5/bn")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv_6",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_6/bn")
        net = tf.nn.relu(net) 
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv_7",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_7/bn")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool3")  
        print ("the 3th stage-shape："+str(net.shape))
        
        # 第四区
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv_8",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_8/bn")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv_9",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_9/bn")
        net = tf.nn.relu(net)  
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv_10",padding='SAME') 
        net = tf_op.batch_normalization(net,"conv_10/bn")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool4")  
        print ("the 4th stage-shape："+str(net.shape))
        
        # 第五区
        net_shape = net.shape
        net = tf_op.fc_layer(net, int(net_shape[1])*int(net_shape[2])*int(net_shape[3]), 128, "fc5")
        net = tf_op.drop_out(net,dropout_rate=self.dropout_rate)
        net = tf.nn.relu(net)
        net = tf_op.fc_layer(net, 128, self.class_num, "fc6")
        result = tf.nn.softmax(net,name="prob") 
        print ("the 5th stage-shape："+str(result.shape))
        return result
       