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

class MobileNetV1:
    """
    mobilenet-v1：https://arxiv.org/pdf/1704.04861.pdf
    """
    def __init__(self,class_num,width_multiplier=1,dropout_rate=0.2):
        """
        parameters:
           class_num:the final class number
           dropout_rate:dropout rate
           width_multiplier:the width_multiplier of mobilenet
        """ 
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        self.width_multiplier = width_multiplier
        
        # construct placeholder
        self.inputs = tf.placeholder(tf.float32, [None, 224, 224, 3],name="inputs")  # the input of the network
        self.labels = tf.placeholder(tf.float32, [None, self.class_num],name="labels")  # the labels of train sampels
        self.is_training = tf.placeholder(tf.bool,name="is_training")  # trainable or not
        
        self.tf_op = tf_fun.tf_fun(self.is_training)
        
        # build the network
        self.prob = self.build(self.inputs)
        
        # construct loss Function
        self.cost = -tf.reduce_mean(self.labels*tf.log(tf.clip_by_value(self.prob,1e-10,1.0)))  

    def build(self,inputs,scope="MobileNetV1"): 
        """
        function:
            build the mobilenet-v1 network
        """
        assert inputs.get_shape().as_list()[1:]==[224,224,3], 'the size of inputs is incorrect!'
        
        # start to build the model
        net = inputs 
        print ("start-shape:"+str(net.shape)) 
        
        net = self.tf_op.conv_layer(net, kernel_num=round(32 * self.width_multiplier), kernel_size=3, stride=2, layer_name="conv_1",padding='SAME') 
        net = self.tf_op.batch_normalization(net,"conv_1/batch_norm")
        net = tf.nn.relu(net)
        net = self.tf_op.depthwise_separable_conv(net,64,self.width_multiplier,strides=1,scopename='conv_ds_2') 
        net = self.tf_op.depthwise_separable_conv(net,128,self.width_multiplier,strides=2,scopename='conv_ds_3')
        net = self.tf_op.depthwise_separable_conv(net,128,self.width_multiplier,strides=1,scopename='conv_ds_4')
        net = self.tf_op.depthwise_separable_conv(net,256,self.width_multiplier,strides=2,scopename='conv_ds_5')
        net = self.tf_op.depthwise_separable_conv(net,256,self.width_multiplier,strides=1,scopename='conv_ds_6')
        net = self.tf_op.depthwise_separable_conv(net,512,self.width_multiplier,strides=2,scopename='conv_ds_7')  
        print ("the 1th stage-shape："+str(net.shape))
        
        net = self.tf_op.depthwise_separable_conv(net,512,self.width_multiplier,strides=1,scopename='conv_ds_8')
        net = self.tf_op.depthwise_separable_conv(net,512,self.width_multiplier,strides=1,scopename='conv_ds_9')
        net = self.tf_op.depthwise_separable_conv(net,512,self.width_multiplier,strides=1,scopename='conv_ds_10')
        net = self.tf_op.depthwise_separable_conv(net,512,self.width_multiplier,strides=1,scopename='conv_ds_11')
        net = self.tf_op.depthwise_separable_conv(net,512,self.width_multiplier,strides=1,scopename='conv_ds_12')
        print ("the 2th stage-shape："+str(net.shape))
        
        net = self.tf_op.depthwise_separable_conv(net,1024,self.width_multiplier,strides=2,scopename='conv_ds_13')
        net = self.tf_op.depthwise_separable_conv(net,1024,self.width_multiplier,strides=1,scopename='conv_ds_14')
        print ("the 3th stage-shape："+str(net.shape))
        
        net = self.tf_op.avg_pool(net, "avg_pool_15", kernel_size=7, stride=1, padding='VALID')
        print ("the 4th stage-shape："+str(net.shape))
        
        net = self.tf_op.fc_layer(net, 1024, 1024, "fc16")
        net = self.tf_op.drop_out(net,dropout_rate=self.dropout_rate)
        net = self.tf_op.batch_normalization(net,"fc16/batch_norm")
        net = tf.nn.relu(net)
        print ("the 5th stage-shape："+str(net.shape))
        
        net = self.tf_op.fc_layer(net, 1024, self.class_num, "fc17") 
        result = tf.nn.softmax(net,name="prob")
        print ("the 6th stage-shape："+str(result.shape))
        return result
                
                
                