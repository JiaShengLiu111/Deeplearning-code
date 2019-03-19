# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题

import os
sys.path.append(os.path.dirname(os.getcwd()))  # add the upper level directory into the pwd
import tf_fun
import tensorflow as tf
import densenet_config as cfg

class densenet:
    """
    densenet
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
        
    def bottleneck_compositeLayer(self,bottom,layer_name):
        """
       function:
           composite layer with bottleneck====>Densenet-B:
           BN->Relu->conv(1×1)->BN->Relu->conv(3×3)
       patameters:
           bottom:the feature map waitting to bottleneck_composite
        """
        net = self.tf_op.batch_normalization(bottom, scope_name=layer_name+"_bn1")
        net = tf.nn.relu(net)
        net = self.tf_op.conv_layer(net, kernel_num=4*cfg.growth_rate, kernel_size=1, stride=1, layer_name=layer_name+"_conv1", padding='SAME')
        net = self.tf_op.batch_normalization(net, scope_name=layer_name+"_bn2")
        net = tf.nn.relu(net)
        result = self.tf_op.conv_layer(net, kernel_num=cfg.growth_rate, kernel_size=3, stride=1, layer_name=layer_name+"_conv2", padding='SAME')
        return result
        
    def compression_transitionLayer(self,bottom,layer_name):
        """
       function:
           transition layer with compression====>Densenet-C:
           BN->conv(1×1)->pool(2×2)
       patameters:
           bottom:the feature map waitting to bottleneck_composite
           layer_name:the name of the layer
        """
        net = self.tf_op.batch_normalization(bottom, scope_name=layer_name+"_bn") 
        net = tf.nn.relu(net)
        channel = int(net.get_shape()[3]) # the channel of bottom
        conv11_channel = int(channel * cfg.compression_theta)
        net = self.tf_op.conv_layer(net, kernel_num=conv11_channel, kernel_size=1, stride=1, layer_name=layer_name+"_conv", padding='SAME')
        net = self.tf_op.max_pool(net, layer_name+"_pool", kernel_size=2, stride=2, padding='SAME')
        return net
        
    def dense_block(self,bottom,layer_num,block_name):
        """
       function:
           desneblock
       parameters:
           bottom:the input tensor of the denseblock
           layer_num:the number of layer in the denseblock
           block_name:the name of the denseblock
        """
        layer_outputList = []
        layer_outputList.append(bottom)
        for i in range(layer_num):
            inputl = self.tf_op.concat(layer_outputList,axis=3)  # the input of the Lth layer in denseblock
            net = self.bottleneck_compositeLayer(inputl,block_name+"_bc"+str(i+1))
            layer_outputList.append(net)
        result = self.tf_op.concat(layer_outputList,axis=3)
        return result
        
    def build(self,inputs):
        """
       build the desnenet network 
        """
        assert inputs.get_shape().as_list()[1:]==[224,224,3], 'the size of inputs is incorrect!'
        
        # start to build the densenet model
        net = inputs
        print "start-shape："+str(inputs.shape)
        
        tf_op = tf_fun.tf_fun(self.is_training)
        net = tf_op.conv_layer(net, kernel_num=2*cfg.growth_rate, kernel_size=7, stride=2, layer_name="conv1",padding='SAME')
        net = self.tf_op.batch_normalization(net,scope_name="bn1")
        net = tf.nn.relu(net) 
        net = tf_op.max_pool(net,layer_name="pool1", kernel_size=3, stride=2, padding='SAME')
        print "the 1th stage-shape："+str(net.shape)
        
        net = self.dense_block(net,6,"denseblock1") 
        net = self.compression_transitionLayer(net,"transition1")
        print "the 2th stage-shape："+str(net.shape)
        
        net = self.dense_block(net,12,"denseblock2") 
        net = self.compression_transitionLayer(net,"transition2")
        print "the 3th stage-shape："+str(net.shape)
        
        net = self.dense_block(net,24,"denseblock3") 
        net = self.compression_transitionLayer(net,"transition3")
        print "the 4th stage-shape："+str(net.shape)
        
        net = self.dense_block(net,16,"denseblock4") 
        net = self.tf_op.batch_normalization(net,scope_name="bn4")
        net = tf.nn.relu(net) 
        net = self.tf_op.global_avg_pool(net, "pool4", stride=1)
        print "the 5th stage-shape："+str(net.shape)
        
        shape = net.get_shape()
        in_size = shape[1]*shape[2]*shape[3]
        net = self.tf_op.fc_layer(net, int(in_size), self.class_num, "fc5") 
        net = tf_op.drop_out(net,dropout_rate=self.dropout_rate)
        result = tf.nn.softmax(net,name="prob")
        print "the 6th stage-shape："+str(result.shape)
        return result 
    
    
    