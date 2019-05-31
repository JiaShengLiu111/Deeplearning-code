# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题


# the parameters of mobilenetv2 network
expansion_ratio = 6  # the expansion ratio of the first conv(1×1) of the bottleneck
width_multiplier = 1  # the width_multiplier of mobilenet