# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题

import os
# sys.path.append(os.path.dirname(os.getcwd()))  # add the upper level directory into the pwd
sys.path.append("./Deeplearning_code")  # add the directory of Deeplearning_code
import tf_fun
import tensorflow as tf

class Ann:
    """
    ann network
    """
    def __init__(self,class_num,dropout_rate=0.2):
        """
        parameters:  
           class_num:the final class number
           dropout_rate:dropout rate
        """ 
        self.class_num = class_num
        self.dropout_rate = dropout_rate 
        
        # construct placeholder
        self.inputs = tf.placeholder(tf.float32,[None,114])  # the input of the network
        self.labels = tf.placeholder(tf.float32,[None,2])  # the labels of train sampels
        self.is_training = tf.placeholder(tf.bool)  # trainable or not
        
        # build the network
        self.prob = self.build(self.inputs)
        
        # construct loss Function
        self.cost = -tf.reduce_mean(self.labels*tf.log(tf.clip_by_value(self.prob,1e-10,1.0)))  
        
    def build(self,inputs):
        """
        build the network:inputs(373)+fc(700)+fc(700)+outputs(2)
        """
        assert inputs.get_shape().as_list()[1:]==[114], 'the size of inputs is incorrect!'
        
        # start to build the model
        net = inputs
        print "start-shape："+str(net.shape) 
        
        tf_op = tf_fun.tf_fun(self.is_training)
        net = tf_op.fc_layer(net,114,250,"fc1")
        # net = tf_op.drop_out(net,dropout_rate=self.dropout_rate)
        # net = tf_op.batch_normalization(net,"bn1")
        # net = tf.nn.relu(net)
        net = tf.nn.relu(net)
        print "the 1th stage-shape："+str(net.shape)
        
        net = tf_op.fc_layer(net,250,2,"fc2")
        return tf.nn.softmax(net)