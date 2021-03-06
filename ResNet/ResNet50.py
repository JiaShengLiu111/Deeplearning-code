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

# import os
# sys.path.append(os.path.dirname(os.getcwd()))  # add the upper level directory into the pwd
import tf_fun
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ResNet50:
    """
    mobilenet-v1：https://arxiv.org/pdf/1704.04861.pdf
    """
    def __init__(self,class_num,dropout_rate=0.2):
        """
        parameters:
           class_num:the final class number
           dropout_rate:dropout rate
           width_multiplier:the width_multiplier of mobilenet
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
        
    def bottleneck(self,bottom,conv11_channel1,conv33_channel,conv11_channel2,block_name,stride=1):
        """
       function:
           the bottleneck of ResNet50
       parameters:
           bottom:the input tensor of the denseblock
           conv11_channel1:the number of the channel of the first conv(1×1)
           conv33_channe:the number of the channel of the conv(3×3)
           conv11_channel2:the number of the channel of the second conv(1×1)
           block_name:the name of the bottleneck
           strides：the stride of the first conv(1×1) of the bottleneck
        """
        net1 = self.tf_op.conv_layer(bottom, conv11_channel1, kernel_size=1, stride=stride, layer_name=block_name+"/conv1",padding='VALID')
        net1 = self.tf_op.batch_normalization(net1, scope_name=block_name+"/bn1")
        net1 = tf.nn.relu(net1)
        
        net2 = self.tf_op.conv_layer(net1, conv33_channel, kernel_size=3, stride=1, layer_name=block_name+"/conv2",padding='SAME')
        net2 = self.tf_op.batch_normalization(net2, scope_name=block_name+"/bn2")
        net2 = tf.nn.relu(net2)
        
        net3 = self.tf_op.conv_layer(net2, conv11_channel2, kernel_size=1, stride=1, layer_name=block_name+"/conv3",padding='VALID')
        net3 = self.tf_op.batch_normalization(net3, scope_name=block_name+"/bn3")
        
        # Ensure that the shape of bottom and net3 are equal 
        net3_channel = net3.get_shape()[-1]
        if bottom.shape[1:4]!=net3.shape[1:4]:
            tmp = self.tf_op.conv_layer(bottom, net3_channel, kernel_size=1, stride=stride, layer_name=block_name+"/conv4",padding='VALID')
        else:
            tmp = bottom
        assert net3.shape[1:4]==tmp.shape[1:4], "net3 and tmp have different shapes！"
        
        # identity
        net4 = tf.add(net3,tmp)
        """
        # 注意：add操作之后不能使用BN，这里BN改变了“identity”分支的分布，影响了信息的传递，在训练的时候会阻碍loss的下降
        参考网址：https://blog.csdn.net/chenyuping333/article/details/82344334
        """
        # net4 = self.tf_op.batch_normalization(net4, scope_name=block_name+"/bn4") 
        net4 = tf.nn.relu(net4)
        return net4  

    def build(self,inputs,scope="MobileNetV1"): 
        """
        function:
            build the ResNet50 network
        """
        assert inputs.get_shape().as_list()[1:]==[224,224,3], 'the size of inputs is incorrect!'
        
        # start to build the model
        net = inputs 
        print ("start-shape:"+str(net.shape))
        
        net = self.tf_op.conv_layer(net, kernel_num=64, kernel_size=7, stride=2, layer_name="conv1",padding='SAME')
        print ("the 0th stage-shape："+str(net.shape))
        
        net = self.tf_op.max_pool(net, layer_name="pool1", kernel_size=3, stride=2, padding='SAME')
        print ("the 1th stage-shape："+str(net.shape))
        
        # the first ResNet block
        for i in range(3):
            # s = 2 if i==0 else 1
            s = 1  # all the layers in the block with the stride 1
            net = self.bottleneck(net,64,64,256,block_name="bottleneck2_"+str(i),stride=s)
        print ("the 2th stage-shape："+str(net.shape))
        
        # the second ResNet block
        for i in range(4):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,128,128,512,block_name="bottleneck3_"+str(i),stride=s)
        print ("the 3th stage-shape："+str(net.shape))
        
        # the third ResNet block
        for i in range(6):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,256,256,1024,block_name="bottleneck4_"+str(i),stride=s)
        print ("the 4th stage-shape："+str(net.shape))
        
        # the fourth ResNet block
        for i in range(3):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,512,512,2048,block_name="bottleneck5_"+str(i),stride=s)
        print ("the 5th stage-shape："+str(net.shape))
        
        net = self.tf_op.avg_pool(net, "avg_pool6", kernel_size=2, stride=2, padding='VALID')
        net_shape = net.get_shape() 
        net = self.tf_op.fc_layer(net, int(net_shape[1])*int(net_shape[2])*int(net_shape[3]), self.class_num, "fc7")
        result = tf.nn.softmax(net,name="prob")
        print ("the 6th stage-shape："+str(result.shape))
        return result
