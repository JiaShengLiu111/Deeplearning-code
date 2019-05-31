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
import mobilenetv2_config as cfg

class MobileNetV2:
    """
    mobilenet-v2：https://arxiv.org/pdf/1801.04381.pdf
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
        self.inputs = tf.placeholder(tf.float32, [None, 224, 224, 3],name="inputs")  # the input of the network
        self.labels = tf.placeholder(tf.float32, [None, self.class_num],name="labels")  # the labels of train sampels
        self.is_training = tf.placeholder(tf.bool,name="is_training")  # trainable or not
        
        self.tf_op = tf_fun.tf_fun(self.is_training)
        
        # build the network
        self.prob = self.build(self.inputs)
        
        # construct loss Function
        self.cost = -tf.reduce_mean(self.labels*tf.log(tf.clip_by_value(self.prob,1e-10,1.0)))  

    def bottleneck(sef,bottom,expansion_ratio,output_dim,block_name,stride=1):
        """
        function:
            the bottleneck of mobilenetv2:pw->dw->pw&linear
            参考至：https://github.com/neuleaf/MobileNetV2/blob/master/mobilenet_v2.py
        parameters:
            bottom: the input tensor of the bottleneck
            expansion_ratio: the expansion ratio of the first conv(1×1) of the bottleneck
            output_dim: the channel of the output of the bottleneck
            block_name:the name of the bottleneck
            strides：the stride of the dw_conv(3×3) of the bottleneck
        """
        # pw
        channel = int(expansion_ratio*bottom.shape.as_list()[-1])
        net1 = self.tf_op.conv_layer(bottom, channel, kernel_size=1, stride=1, layer_name=block_name+"/pw1",padding='SAME')
        net1 = self.tf_op.batch_normalization(net1, scope_name=block_name+"/bn1")
        net1 = tf.nn.relu(net1)
        
        # dw
        net2 = slim.separable_convolution2d(net1,num_outputs=None,stride=strides,depth_multiplier=1\
                                 ,activation_fn=None,padding='SAME',kernel_size=[3, 3],scope=block_name+'/dw')
        net2 = self.tf_op.batch_normalization(net2, scope_name=block_name+"/bn2")
        net2 = tf.nn.relu(net2)
        
        # pw & linear
        net3 = self.tf_op.conv_layer(net2, output_dim, kernel_size=1, stride=1, layer_name=block_name+"/pw2",padding='SAME')
        net3 = self.tf_op.batch_normalization(net3, scope_name=block_name+"/bn3")
        
        # Ensure that the shape of bottom and net3 are equal 
        net3_channel = net3.get_shape()[-1]
        if bottom.shape[1:4]!=net3.shape[1:4]:
            tmp = self.tf_op.conv_layer(bottom, net3_channel, kernel_size=1, stride=stride, layer_name=block_name+"/pw3",padding='SAME')
        else:
            tmp = bottom
        assert net3.shape[1:4]==tmp.shape[1:4], "net3 and tmp have different shapes！"
        
        # identity
        net4 = tf.add(net3,tmp)
        net4 = self.tf_op.batch_normalization(net4, scope_name=block_name+"/bn3")  # 注意是非线性层，无需添加激活函数。
        return net4 
        
    def build(self,inputs,scope="MobileNetV2"): 
        """
        function:
            build the mobilenet-v2 network
        """
        assert inputs.get_shape().as_list()[1:]==[224,224,3], 'the size of inputs is incorrect!'
        
        # start to build the model
        net = inputs 
        print ("start-shape:"+str(net.shape)) 
        
        net = self.tf_op.conv_layer(net, kernel_num=round(32 * cfg.width_multiplier), kernel_size=3, stride=2, layer_name="conv_1",padding='SAME') 
        net = self.tf_op.batch_normalization(net,"conv_1/batch_norm")
        net = tf.nn.relu(net)
        print ("the 1th stage-shape："+str(net.shape))
        
        # the 1 block of MobileNetV2
        for i range(1):
            # s = 2 if i==0 else 1 
            s = 1  # all the layers in the block with the stride 1
            net = self.bottleneck(net,1,round(16 * cfg.width_multiplier),block_name="bottleneck2_"+str(i),stride=s)
        print ("the 2th stage-shape："+str(net.shape))
        
        # the 2 block of MobileNetV2
        for i range(2):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,cfg.expansion_ratio,round(24 * cfg.width_multiplier),block_name="bottleneck3_"+str(i),stride=s)
        print ("the 3th stage-shape："+str(net.shape))
        
        # the 3 block of MobileNetV2
        for i range(3):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,cfg.expansion_ratio,round(32 * cfg.width_multiplier),block_name="bottleneck4_"+str(i),stride=s)
        print ("the 4th stage-shape："+str(net.shape))
        
        # the 4 block of MobileNetV2
        for i range(4):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,cfg.expansion_ratio,round(64 * cfg.width_multiplier),block_name="bottleneck5_"+str(i),stride=s)
        print ("the 5th stage-shape："+str(net.shape))
        
        # the 5 block of MobileNetV2
        for i range(3):
            # s = 2 if i==0 else 1
            s = 1  # all the layers in the block with the stride 1
            net = self.bottleneck(net,cfg.expansion_ratio,round(96 * cfg.width_multiplier),block_name="bottleneck6_"+str(i),stride=s)
        print ("the 6th stage-shape："+str(net.shape))
        
        # the 6 block of MobileNetV2
        for i range(3):
            s = 2 if i==0 else 1 
            net = self.bottleneck(net,cfg.expansion_ratio,round(160 * cfg.width_multiplier),block_name="bottleneck7_"+str(i),stride=s)
        print ("the 7th stage-shape："+str(net.shape))
        
        # the 7 block of MobileNetV2
        for i range(1):
            # s = 2 if i==0 else 1 
            s = 1  # all the layers in the block with the stride 1
            net = self.bottleneck(net,cfg.expansion_ratio,round(320 * cfg.width_multiplier),block_name="bottleneck8_"+str(i),stride=s)
        print ("the 8th stage-shape："+str(net.shape))
        
        net = self.tf_op.conv_layer(net, kernel_num=round(1280 * cfg.width_multiplier), kernel_size=1, stride=1, layer_name="conv_2",padding='SAME') 
        net = self.tf_op.global_avg_pool(net, "global_avg_pool", stride=1)
        net = self.tf_op.conv_layer(net, kernel_num=self.class_num, kernel_size=1, stride=1, layer_name="conv_3",padding='SAME') 
        net = tf.reshape(net, [-1, self.class_num])
        result = tf.nn.softmax(net,name="prob")
        print ("the 9th stage-shape："+str(result.shape))
        return result
                
              