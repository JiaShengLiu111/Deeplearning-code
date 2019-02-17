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

class Vgg19:
    """
    vgg19
    """
    def __init__(self,inputs,is_training,class_num,dropout_rate=0.2):
        """
        parameters:
           inputs:the input of the network
           is_training:trainable or not
           class_num:the final class number
           dropout_rate:dropout rate
        """
        self.is_training=is_training
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        self.prob = self.build(inputs)
        
    def build(self,inputs):
        assert inputs.get_shape().as_list()[1:]==[224,224,3], 'the size of inputs is incorrect!'
        
        # start to build the model
        net = inputs
        print "start-shape："+str(inputs.shape)
        
        tf_op = tf_fun.tf_fun(self.is_training)
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv1_1",padding='SAME')
        net = tf_op.batch_normalization(net,"bn1_1")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=64, kernel_size=3, stride=1, layer_name="conv1_2",padding='SAME')
        net = tf_op.batch_normalization(net,"bn1_2")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool1")
        print "the 1th stage-shape："+str(net.shape)
        
        net = tf_op.conv_layer(net, kernel_num=128, kernel_size=3, stride=1, layer_name="conv2_1",padding='SAME')
        net = tf_op.batch_normalization(net,"bn2_1")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=128, kernel_size=3, stride=1, layer_name="conv2_2",padding='SAME')
        net = tf_op.batch_normalization(net,"bn2_2")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool2")
        print "the 2th stage-shape："+str(net.shape)
        
        net = tf_op.conv_layer(net, kernel_num=256, kernel_size=3, stride=1, layer_name="conv3_1",padding='SAME')
        net = tf_op.batch_normalization(net,"bn3_1")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=256, kernel_size=3, stride=1, layer_name="conv3_2",padding='SAME')
        net = tf_op.batch_normalization(net,"bn3_2")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=256, kernel_size=3, stride=1, layer_name="conv3_3",padding='SAME')
        net = tf_op.batch_normalization(net,"bn3_2")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=256, kernel_size=3, stride=1, layer_name="conv3_4",padding='SAME')
        net = tf_op.batch_normalization(net,"bn3_4")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool3")
        print "the 3th stage-shape："+str(net.shape)
        
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv4_1",padding='SAME')
        net = tf_op.batch_normalization(net,"bn4_1")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv4_2",padding='SAME')
        net = tf_op.batch_normalization(net,"bn4_2")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv4_3",padding='SAME')
        net = tf_op.batch_normalization(net,"bn4_3")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv4_4",padding='SAME')
        net = tf_op.batch_normalization(net,"bn4_4")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool4")
        print "the 4th stage-shape："+str(net.shape)
        
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv5_1",padding='SAME')
        net = tf_op.batch_normalization(net,"bn5_1")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv5_2",padding='SAME')
        net = tf_op.batch_normalization(net,"bn5_2")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv5_3",padding='SAME')
        net = tf_op.batch_normalization(net,"bn5_3")
        net = tf.nn.relu(net)
        net = tf_op.conv_layer(net, kernel_num=512, kernel_size=3, stride=1, layer_name="conv5_4",padding='SAME')
        net = tf_op.batch_normalization(net,"bn5_4")
        net = tf.nn.relu(net)
        net = tf_op.max_pool(net,layer_name="pool5")
        print "the 5th stage-shape："+str(net.shape)
        
        net = tf_op.fc_layer(net, 7*7*512, 4096, "fc6")
        net = tf_op.drop_out(net,dropout_rate=self.dropout_rate)
        net = tf.nn.relu(net)
        net = tf_op.fc_layer(net, 4096, 4096, "fc7")
        net = tf_op.drop_out(net,dropout_rate=self.dropout_rate)
        net = tf.nn.relu(net)
        net = tf_op.fc_layer(net, 4096, self.class_num, "fc8")
        result = tf.nn.softmax(net) 
        print "the 6th stage-shape："+str(result.shape)
        return result
        
        
        
        