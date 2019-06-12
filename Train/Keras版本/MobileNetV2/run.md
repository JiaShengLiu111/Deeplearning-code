训练代码如下所示：

```
import time
import Build
import tensorflow as tf  
from IPython.display import display_html
from keras import backend as K
import Config as cfg0

def trainFun(model,utils,X_train,y_train,X_train_evalua,y_train_evalua,dataenhance,reDir='info/info.txt'):
    time_total = 0
    utils.printRd("",reDir,"w+")  # 清空重定向文件 
    # tf.reset_default_graph()    # 清空运行图
    batch_size = cfg0.batch_size        

    # 表示训练多少个batch对应一epoch 
    epoch = int(len(X_train)/batch_size+1e-9)  
    # 设置训练轮数
    STEPS = int(epoch*cfg0.epochs)  
    print "the number of total batch:"+str(STEPS)

    with tf.Session(config = config) as sess:    
        sess.run(tf.global_variables_initializer())  # 初始化变量  
        
        # 判断是否加载与训练模型
        if cfg0.loadmodel_path!=None:
            model = load_model(cfg0.loadmodel_path+cfg0.loadmodel_name)

        val_accuracy_list = []  # 绘制验证集准确率变化曲线
        val_f1_list = []  # 绘制验证集f1_score变化曲线
        train_accuracy_list = []  # 绘制训练集准确率变化曲线
        cost_list_train = []
        cost_list_val = []

        val_f_beta_score_max = -1 # 记录val_f1的最大值
        val_f1_max = -1 # 记录val_f1的最大值
        for i in range(0,STEPS):   # 表示从索引为start开始训练样本（主要是为了均衡二次训练时样本被训练的机会）
            utils.printRd("............................第"+str(i)+"个batch............................",reDir)  
            # 首先进行学习率指数衰减计算
            lr_decay_step = int(STEPS/(cfg0.lr_decay_times+1e-9))
            model = utils_.lr_exponential_decay(model,i,lr_decay_step,cfg0.lr_decay_rate)
            if i%epoch==0:  # 每一个epoch开始时，将训练重新打乱(有正则化作用)
                X_train = utils.myShuffle(X_train,666)
                y_train = utils.myShuffle(y_train,666)
            
            if i%(int(cfg0.saveStepEpochRate*epoch))==0 and (i!=0 or cfg0.firstPred==True):  # 判断在进行训练之前是否进行预测。 
                # 记录“训练集准确率”、“验证集准确率”、“验证集F1”  
                train_predicts,_ = utils.predict_keras(model,X_train_evalua,batch_size,dataenhance,reDir)  
                val_predicts,_ = utils.predict_keras(model,X_val,batch_size,dataenhance,reDir)
                
                # 求真实标签
                train_evalua_realLabel = utils.onehot2realLabel(y_train_evalua)
                train_predicts_realLabel = utils.onehot2realLabel(train_predicts,utils.countSample(X_train,labels)) 
                
                val_realLabel = utils.onehot2realLabel(y_val)
                val_predicts_realLabel = utils.onehot2realLabel(val_predicts,utils.countSample(X_train,labels))

                # 计算“准确率”等 
                train_accuracy = accuracy_score(train_evalua_realLabel,train_predicts_realLabel, normalize=True)
                val_accuracy = accuracy_score(val_realLabel,val_predicts_realLabel, normalize=True)
                val_f1 = f1_score(val_realLabel,val_predicts_realLabel,average="macro") 
                train_confusion_matrix = confusion_matrix(train_evalua_realLabel,train_predicts_realLabel)
                val_confusion_matrix = confusion_matrix(val_realLabel,val_predicts_realLabel)
                
                # 修改模型评价指标为“异常类别”的f_beta_score
                val_f_beta_score = F_beta_score_MergeNormalAndJunk(val_confusion_matrix,beta=1)  # beta=2，表示“异常类别”查全率的重要性是查准率的4倍
                
                # 训练集交叉熵
                cross_entropy_train = sess.run(-tf.reduce_mean(y_train_evalua*tf.log(tf.clip_by_value(train_predicts,1e-10,1.0))))
                cross_entropy_val = sess.run(-tf.reduce_mean(y_val*tf.log(tf.clip_by_value(val_predicts,1e-10,1.0))))
                
                # 对各个参数只保留五位小数（后面的位四舍五入处理）
                train_accuracy = round(train_accuracy,5)
                val_accuracy = round(val_accuracy,5)
                cross_entropy_train = round(cross_entropy_train,5)
                cross_entropy_val = round(cross_entropy_val,5)
                val_f1 = round(val_f1,5) 
                val_f_beta_score = round(val_f_beta_score,5)

                utils.printRd("训练集混淆矩阵为：",reDir)
                utils.printRd(np.array(train_confusion_matrix),reDir)
                utils.printRd(analysisConfusionMatrix(train_confusion_matrix),reDir)
                utils.printRd("验证集混淆矩阵为：",reDir)
                utils.printRd(np.array(val_confusion_matrix),reDir)
                utils.printRd(analysisConfusionMatrix(val_confusion_matrix),reDir)

                # 计算模型的训练速度
                speed = 1000*time_total/(i+1e-9)  # 训练一个batch的平均时长
                speed = speed/batch_size  # 训练一个样本的平均时长
                speed = round(speed,4)
                utils.printRd("模型训练一个样本的平均时长为："+str(speed)+"毫秒",reDir)
                utils.printRd("第"+str(i)+"次迭代："+"train_accuracy="+str(train_accuracy)+" val_accuracy="+str(val_accuracy)+" cost="+str(cross_entropy_train)+" val_f1:"+str(val_f1)+ " val_f_beta_score:"+str(val_f_beta_score),reDir) 
 
                train_accuracy_list.append(train_accuracy)
                val_accuracy_list.append(val_accuracy)
                val_f1_list.append(val_f1) 
                cost_list_train.append(cross_entropy_train)
                cost_list_val.append(cross_entropy_val)
                lists = [val_accuracy_list,val_f1_list,train_accuracy_list]
                names = ["val_accuracy","val_f1","train_accuracy"]
                utils.showPerformOrCostCurve(lists,names,os.path.dirname(reDir)+'/perform.jpg') # 绘制准确率/F1分数
                utils.writeCurveValue(lists,names,os.path.dirname(reDir)+'/perform.txt') # 将准确率/F1分数列表写入文件
                lists = [cost_list_train,cost_list_val]
                names = ["train_cost","val_cost"]
                utils.showPerformOrCostCurve(lists,names,os.path.dirname(reDir)+'/cost.jpg') # 绘制损失函数
                utils.writeCurveValue(lists,names,os.path.dirname(reDir)+'/cost.txt') # 将损失函数列表写入文件
                
                if val_f1>val_f1_max:
                    val_f1_max = val_f1
                    utils.printRd("val_f1最大值更新："+str(val_f1_max),reDir)

                # 修改模型保存条件为：每次epoch判断一次val_f1是否大于val_f1_max,假如满足条件就保存一次模型
                if val_f_beta_score>val_f_beta_score_max:  # 这样做也就相当于earlystoping。
                    # vgg.save_npy(sess,pwd+'jsL/keras_model/vggPara-save_train_l2正则化解决过拟合.npy')     
                    # 因为模型中使用了BN层，参数无法提取，所以替换使用Saver.save函数保存模型
                    # saver.save(sess, cfg0.saveModelPath)  # 保存模型  
                    model.save(cfg0.saveModelPath)
                    val_f_beta_score_max = val_f_beta_score
                    utils.printRd("model saved!",reDir)
                else:
                    utils.printRd("model not saved!",reDir)  

            # 每次选取batch_size个样本进行训练
            # 梯度下降训练模型参数
            start = (i*batch_size)%len(X_train)
            end = min(start+batch_size,len(X_train)) 
            xxx = dataenhance.getMiniBatch4Train(X_train[start:end])
            # sess.run(train, feed_dict={model.inputs: xxx, model.labels: y_train[start:end], model.is_training: True})  
            yyy = np.array(y_train[start:end])
            time_start = time.time()  # 起始时间
            model.train_on_batch(xxx,yyy,class_weight=None, sample_weight=None)
            time_end = time.time()  # 结束时间
            time_total = time_total + time_end - time_start  # 累加模型训练的时间 
        display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
        sess.close() 
```

调用方式如下所示：

```
reDir = cfg0.train_reDir
dataenhance = utils.DataEnhance(image_h = 224,image_w = 224)
model = Build.MobileNet_V2().build()
trainFun(model,utils_,X_train,y_train,X_train_evalua,y_train_evalua,dataenhance,reDir)
```

