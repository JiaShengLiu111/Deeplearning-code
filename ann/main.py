# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # Solving the problem of Chinese character coding

import os
sys.path.append(os.path.dirname(os.getcwd())) # add the upper level directory into the pwd

import loaddata
import preprocessdata
import annet
import utils 
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score   
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score  
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np

# 训练网络 
import tensorflow as tf   

def trainFun(start=0,batch_size=64,weight_decay=1e-8,lr=1e-4\
             ,epochs=80,firstPred=True,modelPath='jsL/keras_model/'\
             ,loadmodel=None,modelname=None,reDir='info'):
    """
    function for train
    """
    # 首先情况重定向文件
    f=open(reDir+'/info.txt','w+')
    print >> f,"" 
    f.close()
    # 清空运行图
    tf.reset_default_graph()  
    batch_size = batch_size 
    # 实例化网络模型
    net = annet.Ann(class_num=2)  
    # 表示训练多少个batch对应一epoch 
    epoch = int(len(X_train)/batch_size+0.00000001)  
    # 配置tensorflow实用显存的方式
    config = tf.ConfigProto(allow_soft_placement=True)  # 自动选择可以运行的设备 
    config.gpu_options.allow_growth = True  # Gpu内存按需增长 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3 
    saver = tf.train.Saver(max_to_keep=1) # 模型保存器 
    
    # 使用SGD优化器，并且使用权重衰减
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1,global_step,5000,0.96,staircase=True)
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(net.cost,global_step=global_step)

    with tf.Session(config = config) as sess:    
        # 初始化所有变量
        sess.run(tf.global_variables_initializer())   
        # 判断是否加载与训练模型
        if loadmodel!=None:
            saver = tf.train.import_meta_graph(loadmodel+modelname)
            saver.restore(sess,tf.train.latest_checkpoint(loadmodel))
        # 准确率值记录
        test_accuracy_list = []  # 绘制验证集准确率变化曲线
        test_f1_list = []  # 绘制验证集f1_score变化曲线
        train_accuracy_list = []  # 绘制训练集准确率变化曲线
        cost_list = []
        # best score
        test_f_beta_score_max = -1  
        test_f1_max = -1  
        # the total number of train epoch
        STEPS = epoch*epochs  
        print "STEPS:"+str(STEPS) 
        for i in range(start,STEPS+start):   # 表示从索引为start开始训练样本（主要是为了均衡二次训练时样本被训练的机会）
            utils.printRd("............................第"+str(i)+"个batch............................",reDir)  
            if i%(int(1.0*epoch))==0 and (i!=0 or firstPred==True):  # 判断在进行训练之前是否进行预测。 
                
                # 得到X_train,X_test的预测结果
                train_predicts,_ = utils.myPredicts(sess,net.prob,net.inputs,net.is_training,X_train,batch_size,reDir)
                test_predicts,_ = utils.myPredicts(sess,net.prob,net.inputs,net.is_training,X_test,batch_size,reDir)
                # 求真实标签
                train_realLabel = utils.getRealLabel(y_train)
                train_predictLabel = utils.getRealLabel(train_predicts)  
                test_realLabel = utils.getRealLabel(y_test)
                test_predictLabel = utils.getRealLabel(test_predicts)

                # 计算“准确率”等
                train_accuracy = accuracy_score(train_realLabel,train_predictLabel, normalize=True)
                test_accuracy = accuracy_score(test_realLabel,test_predictLabel, normalize=True)
                test_f1 = f1_score(test_realLabel,test_predictLabel,average="macro") 
                train_confusion_matrix = confusion_matrix(train_realLabel,train_predictLabel)
                test_confusion_matrix = confusion_matrix(test_realLabel,test_predictLabel)
                # 训练集交叉熵
                cross_entropy = sess.run(-tf.reduce_mean(y_train*tf.log(tf.clip_by_value(train_predicts,1e-10,1.0))))
                
                # 对各个参数只保留五位小数（后面的位四舍五入处理）
                train_accuracy = round(train_accuracy,5)
                test_accuracy = round(test_accuracy,5)
                cross_entropy = round(cross_entropy,5)
                test_f1 = round(test_f1,5) 

                utils.printRd("训练集混淆矩阵为：",reDir)
                utils.printRd(np.array(train_confusion_matrix),reDir) 
                utils.printRd("验证集混淆矩阵为：",reDir)
                utils.printRd(np.array(test_confusion_matrix),reDir)  
                utils.printRd("第"+str(i)+"次迭代："+"train_accuracy="+str(train_accuracy)+\
                              " test_accuracy="+str(test_accuracy)+" cost="+str(cross_entropy)\
                              +" test_f1:"+str(test_f1),reDir)
 
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
                test_f1_list.append(test_f1) 
                cost_list.append(cross_entropy)
                utils.showPerformCurve(test_accuracy_list,test_f1_list,train_accuracy_list,reDir+'/perform.jpg') # 显示曲线 
                utils.showPerformCurve_cost(cost_list,reDir+'/cost.jpg') # 显示曲线 
                
                if test_f1>test_f1_max:
                    test_f1_max = test_f1
                    saver.save(sess, modelPath)  # 保存模型 
                    utils.printRd("model saved!",reDir)
                else:
                    utils.printRd("model not saved!",reDir)  

            # 每次选取batch_size个样本进行训练
            # 梯度下降训练模型参数
            start = (i*batch_size)%len(X_train)
            end = min(start+batch_size,len(X_train)) 
            xxx = X_train[start:end]
            sess.run(train, feed_dict={net.inputs: xxx, net.labels: y_train[start:end], net.is_training: True})   
        sess.close()

def main():
    # train datas
    loadmodel = None
    modelname = None
    reDir = 'ann_info'   
    trainFun(0,batch_size=256,weight_decay=None,lr=1e-2\
             ,epochs=800,firstPred=True,modelPath=modelPath,loadmodel=loadmodel\
             ,modelname=modelname,reDir=reDir)
# load argv
negative = sys.argv[1]
positive = sys.argv[2]
testneg = sys.argv[3]
testpos = sys.argv[4] 
modelPath = sys.argv[5]
# load datas
Datas_tmp = loaddata.Datas(negative,positive,testneg,testpos)
X_train,y_train,X_test,y_test= Datas_tmp.getDatas()
# preprocess datas
PrepDatas_tmp = preprocessdata.PrepDatas(X_train,X_test)
X_train,X_test = PrepDatas_tmp.getDatas()    

if __name__ == "__main__":
    main()
