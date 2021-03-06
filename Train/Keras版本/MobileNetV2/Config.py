# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题


# the parameters of regular
weight_decay_rate = 1e-6  # L2正则化权重衰减系数
dropout_rate = 0.2  

# the parameters of train process
batch_size = 64
epochs = 40  # the total epochs of the total train_set
firstPred = True  # 表示在首次训练之前是否计算模型初始的损失函数
saveModelPath = '/notebooks/17_LJS/model2019/20190525_TBS三分类加入淡染样本/MobileNetV2/model_1'  # 模型的保存路径
train_reDir = "info/info.txt"  # 训练过程输出信息重定向文件
test_redir = "info_test/info.txt"  # 预测过程输出信息重定向文件

# !!！注意!!！：一旦loadmodel_path则会加载预训练模型
loadmodel_path = None  # 预训练模型路径
loadmodel_name = None  # 预训练模型名称
saveStepEpochRate = 0.07  # 表示每训练saveStepEpochRate*epoch时，判断并保存一次最优模型

# the parameters of ptimizer
lr = 1e-3  # 初始学习率
lr_decay_times = 4  # （训练过程中）学习率衰减次数
lr_decay_rate = 0.94  # 学习率指数衰减率

  

