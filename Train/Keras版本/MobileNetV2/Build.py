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


from keras.applications.mobilenet_v2 import MobileNetV2 as mobilenet_v2 
from keras.layers import Input 
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import Adagrad
from keras.optimizers import SGD
import Config as cfg0

class MobileNet_V2():
    """
    Keras MobileNetV2
    """
    def __init__(self):
        pass
    
    def build(self):
        """
       定义模型网络结构 
        """
        # 定义模型
        # 导入MobileNetV2基模型 
        base_model = mobilenet_v2(weights=None, input_shape=(224, 224, 3),include_top=False)  # 加载模型（放弃后三层全连接参数）
        # 基于base_model构建新模型
        x = base_model.output  # 基础网络的输出
        x = GlobalAveragePooling2D()(x)   
        predictions = Dense(3, activation='softmax')(x)
        inputs = Input(shape=(224,224,3))
        model = Model(inputs=base_model.input, outputs=predictions)  # 根据基础网络新建一个新网络 

        # 编译模型
        sgd = SGD(lr=cfg0.lr, decay=cfg0.weight_decay_rate, momentum=0.9, nesterov=True) 
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  # 编译模型（注意：此处指定交叉熵作为模型损失函数）
        return model








