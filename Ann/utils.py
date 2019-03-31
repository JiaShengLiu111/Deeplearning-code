# _*_ coding:utf-8 _*_
import skimage
import skimage.io
import skimage.transform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image
import time

# synset = [l.strip() for l in open('synset.txt').readlines()]

# 定义可重定向的print函数
def printRd(value,filepath='info'):  # filepath：表示重定向输出文件路径；value：表示输出内容
    f=open(filepath+'/info.txt','a+')
    print >> f,value 
    f.close()


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

# 解决python写入图片后读取图片，图片通道数变成4的问题
def load_image1(path):
    # load image
    img = skimage.io.imread(path)
    img = img[:,:,0:3]
    
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

# 加载训练数据，同时对数据进行增强。（注意：该函数是针对训练数据）
def loadAndEnhanceTrainData(path): 
    # img = cv2.imread(path) # 读取图片
    img = Image.open(path) 
    # img = img[:,:,0:3] # 防止图片变成四通道
    
    img = dataEnhance(img,0.8) # 对图片进行数据增强（随机截取0.8的边长，随机水平翻转，随机竖直翻转）
    img = np.array(img)
    img = img / 255.0 # 对图片进行最大值归一化
    assert (0 <= img).all() and (img <= 1.0).all()
    
    resized_img = skimage.transform.resize(img, (128, 128)) # resize图片大小:resize to 128, 128
    return resized_img

# 加载测试数据，同时对数据进行增强。（注意：该函数是针对测试数据，该函数并不存在随机的部分）
def loadAndEnhanceTestData(path): 
    # img = cv2.imread(path) # 读取图片 
    img = Image.open(path) 
    # img = img[:,:,0:3] # 防止图片变成四通道
    
    img = cropImageOfRate(img,0.8) # 根据rate从图片的中心截取一个ROI图像
    img = np.array(img)
    img = img / 255.0 # 对图片进行最大值归一化
    assert (0 <= img).all() and (img <= 1.0).all()
    
    resized_img = skimage.transform.resize(img, (128, 128)) # resize图片大小:resize to 128, 128
    return resized_img
    


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    # top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    # print(("Top5: ", top5))
    return top1

# returns the real label
# prob_tmp[0]:垃圾类别；prob_tmp[1]:阴性类别；prob_tmp[2]：阳性类别。
def getRealLabel(prob):
    result = []
    for i in range(len(prob)):
        prob_tmp = prob[i] 
        
        #         # 三分类
        #         value = list(prob_tmp)
        #         value_max = max(value)

        #         if value[2]==value_max:  # 注意：首先处理“阳性类别”，从而使得模型具有较低的“假阴性”。
        #             result.append(2)
        #         elif value[0]==value_max:
        #             result.append(0)
        #         else:
        #             result.append(1)
        
        # 二分类
        if prob_tmp[0]>prob_tmp[1]:
            result.append(0)
        else:
            result.append(1)
    result = np.array(result)
    return result

# returns the real label
# prob_tmp[0]:垃圾类别；prob_tmp[1]:阴性类别；prob_tmp[2]：阳性类别。
def getRealLabel2(prob,dataNum):  # dataNum是包含三个元素的一维数组,三个分别表示训练集中各类（垃圾、阴性、阳性）样本的数量
    result = []
    for i in range(len(prob)):
        prob_tmp = prob[i] 
        value = list(prob_tmp)
        value[0] = value[0]/dataNum[0] * sum(dataNum)
        value[1] = value[1]/dataNum[1] * sum(dataNum)
        value[2] = value[2]/dataNum[2] * sum(dataNum)
        
        value_max = max(value)
        
        if value[2]==value_max:  # 注意：首先处理“阳性类别”，从而使得模型具有较低的“假阴性”。
            result.append(2)
        elif value[0]==value_max:
            result.append(0)
        else:
            result.append(1)
        
        # 原始二分类的代码
        #         if prob_tmp[0]>prob_tmp[1]:
        #             result.append(0)
        #         else:
        #             result.append(1)
    result = np.array(result)
    return result

# show performanceCurve
def showPerformCurve(val_accuracy_list,val_f1_list,train_accuracy_list,picFullPath):
    # 参考网址：https://blog.csdn.net/whjstudy1/article/details/80484613
     # 绘制验证集数据准确率变化图
    val_accuracy_index = [i for i in range(len(val_accuracy_list))]
    plt.plot(val_accuracy_index,val_accuracy_list,linewidth=2,label="val_accuracy")
    # 绘制验证集数据f1_score变化图
    val_f1_index = [i for i in range(len(val_f1_list))]
    plt.plot(val_f1_index,val_f1_list,linewidth=2,label="val_f1") 
    # 绘制训练集数据准确率变化图
    train_accuracy_index = [i for i in range(len(train_accuracy_list))]
    plt.plot(train_accuracy_index,train_accuracy_list,linewidth=2,label="train_accuracy")
    plt.legend(loc="best")
    
    plt.title("performance curve",fontsize=24)
    # plt.xlabel("2epoch",fontsize=14)
    # plt.ylabel("accuracy",fontsize=14)
    
    
    # plt.show()
    # 修改为不显示图片，而是保存图片
    plt.savefig(picFullPath)
    plt.clf()  
    plt.close()
    
 # show performanceCurve
def showPerformCurve_cost(cost_list,picFullPath):
    # 参考网址：https://blog.csdn.net/whjstudy1/article/details/80484613 
    # 绘制损失函数变化图
    cost_index = [i for i in range(len(cost_list))]
    plt.plot(cost_index,cost_list,linewidth=2,label="cost")
    
    plt.legend(loc="best") 
    plt.title("cost curve",fontsize=24)
    # plt.xlabel("2epoch",fontsize=14)
    # plt.ylabel("accuracy",fontsize=14)
    
    
    # plt.show()
    # 修改为不显示图片，而是保存图片
    plt.savefig(picFullPath)
    plt.clf()  
    plt.close()
    

# 根据某个minibatch的路径读取minibatch数据
def getMiniBatch4TestAndVal(batchDir): 
    allSamples = []
    allSamplesDir = batchDir
    for i in range(len(allSamplesDir)):
        img_tmp = loadAndEnhanceTestData(allSamplesDir[i])
        allSamples.append(img_tmp) 
    allSamples = np.array(allSamples) 
    return allSamples

# 根据某个minibatch的路径读取minibatch数据
def getMiniBatch4Train(batchDir):
    allSamples = []
    allSamplesDir = batchDir
    for i in range(len(allSamplesDir)):
        img_tmp = loadAndEnhanceTrainData(allSamplesDir[i])
        allSamples.append(img_tmp) 
    allSamples = np.array(allSamples)
    return allSamples


    
# 输入：特征向量集合
# 输出：预测结果
def myPredicts(sess,sess_op,images,train_mode,X_input,batch_size,reDir):
    ReadDataTime = 0  # 用于记录访问磁盘读取数据的时间
    time_start = time.time()  # 起始时间
    result = []
    for i in range(0,len(X_input),batch_size):
        if i%100==0:
            printRd("myPredicts:"+str(i),reDir)
        start = i
        end = min(start+batch_size,len(X_input)) 
        
        read_start = time.time()
        XX = X_input[start:end]  # 读取数据
        read_end = time.time()
        read_time = read_end - read_start
        ReadDataTime = ReadDataTime + read_time  # 累计读取磁盘数据的时间
        
        result_tmp = sess.run(sess_op,feed_dict={images:XX,train_mode:False})
        # print("result_tmp:")
        # print result_tmp
        result_tmp = list(result_tmp)
        result = result+result_tmp 
    time_end = time.time()
    return np.array(result),time_end-time_start-ReadDataTime

# 输入：特征向量集合
# 输出：预测结果
def myPredicts_dense(sess,sess_op,images,true_out,learning_rate,train_mode,X_input,y_input,epoch_learning_rate,batch_size):
    result = []
    for i in range(0,len(X_input),batch_size):
        if i%100==0:
            print("myPredicts:"+str(i))
        start = i
        end = min(start+batch_size,len(X_input)) 
        
        train_feed_dict = {
                images: X_input[start:end],
                true_out: y_input[start:end],
                learning_rate: epoch_learning_rate,
                train_mode : False
            }
        result_tmp = sess.run(sess_op,feed_dict=train_feed_dict)
        # print("result_tmp:")
        # print result_tmp
        result_tmp = list(result_tmp)
        result = result+result_tmp   
    return np.array(result)

def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)
    
    
    
# //////////////////////////////////////////////////////////////////////////////////////////
# 数据增强函数,主要参考网址：https://blog.csdn.net/sinat_29957455/article/details/80629098

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt 
plt.rcParams["font.sans-serif"]=["SimHei"]  #用来正常显示中文
 
# 随机截取函数
def randomCrop2(img,rate):    
    resizeLength = int(img.shape[0]*rate) 
    crop_img = tf.random_crop(img,[resizeLength,resizeLength,3])   #将图片进行随机裁剪为224×224
    # sess = tf.InteractiveSession() 
    crop_img = cv2.cvtColor(crop_img.eval(),cv2.COLOR_BGR2RGB) 
    return crop_img

# 随机水平翻转
def randomFlipLeftRight2(img):    
    h_flip_img = tf.image.random_flip_left_right(img)  #将图片随机进行水平翻转  
    # sess = tf.InteractiveSession()
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #通道转换
    h_flip_img = cv2.cvtColor(h_flip_img.eval(),cv2.COLOR_BGR2RGB)  # 水平翻转 
    return h_flip_img 

# 随机竖直翻转
def randomFlipUpDown2(img):    
    v_flip_img = tf.image.random_flip_up_down(img)  #将图片随机进行垂直翻转  
    # sess = tf.InteractiveSession()
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #通道转换
    v_flip_img = cv2.cvtColor(v_flip_img.eval(),cv2.COLOR_BGR2RGB)  # 水平翻转 
    return v_flip_img  

# 针对单个图片的_增强函数：对输入图片进行随机截取、随机水平翻转、随机竖直翻转
def dataEnhance2(img,rate):
    x = randomCrop(img,rate)  # 随机截取
    x = randomFlipLeftRight(x)  # 随机水平翻转
    x = randomFlipUpDown(x)  # 随机垂直翻转
    return x

# 针对某个batch的_增强函数：对输入图片进行随机截取、随机水平翻转、随机竖直翻转
def batchEnhance2(batch,rate):
    result = []
    for i in range(len(batch)):
        value = dataEnhance(batch[i])
        result.append(value)
    result = np.array(result)
    return result  

# 截取一张图片最中间的ROI图，边长是原始边长的rate倍
def cropImageOfRate2(img,rate):
    resizeLength = int(img.shape[0]*rate)
    yy = int(img.shape[0]*(1-rate)/2)
    xx = yy
    crop_img = img[yy: yy + resizeLength, xx: xx + resizeLength]
    return crop_img
# //////////////////////////////////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////////////////////////////////
# python数据增强函数
# 主要参考网址：https://www.jb51.net/article/45653.htm（产生随机数）
# 网址2：https://blog.csdn.net/guduruyu/article/details/70842142
# 网址3：https://blog.csdn.net/qq_23301703/article/details/79908988

import random
from PIL import Image

# 随机截取函数
def randomCrop(img,rate):     
    length = np.array(img).shape[0]
    resizeLength = int(length*rate) # 结果图片的边长
    rangeLength = length - resizeLength  # 原图和crop边长差
    cropX = random.randint(0, rangeLength)
    cropY = cropX 
    box = (cropX, cropY, cropX+resizeLength, cropY+resizeLength)  # 结果图片的范围
    crop_img = img.crop(box)   #图像裁剪
    # crop_img = np.array(crop_img)
    return crop_img

# 随机水平翻转
def randomFlipLeftRight(img):    
    randNum = random.randint(0, 1)
    if randNum%2==0: 
        h_flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)   # 进行水平翻转
        # h_flip_img = np.array(h_flip_img) 
    else:
        # h_flip_img = np.array(img)
        h_flip_img = img
    return h_flip_img 

# 随机竖直翻转
def randomFlipUpDown(img):   
    randNum = random.randint(0, 1)
    if randNum%2==0: 
        v_flip_img = img.transpose(Image.FLIP_TOP_BOTTOM)   # 进行水平翻转
        # v_flip_img = np.array(v_flip_img) 
    else:
        # v_flip_img = np.array(img)
        v_flip_img = img
    return v_flip_img  

# 针对单个图片的_增强函数：对输入图片进行随机截取、随机水平翻转、随机竖直翻转
def dataEnhance(img,rate):
    x = randomCrop(img,rate)  # 随机截取
    x = randomFlipLeftRight(x)  # 随机水平翻转
    x = randomFlipUpDown(x)  # 随机垂直翻转
    return x

# 针对某个batch的_增强函数：对输入图片进行随机截取、随机水平翻转、随机竖直翻转
def batchEnhance(batch,rate):
    result = []
    for i in range(len(batch)):
        value = dataEnhance(batch[i])
        result.append(value)
    result = np.array(result)
    return result  

# 截取一张图片最中间的ROI图，边长是原始边长的rate倍
def cropImageOfRate(img,rate):
    length = np.array(img).shape[0]
    resizeLength = int(length*rate)
    yy = int(length*(1-rate)/2)
    xx = yy
    # crop_img = img[yy: yy + resizeLength, xx: xx + resizeLength]
    box = (yy,xx,yy+resizeLength,xx+resizeLength)
    crop_img = img.crop(box)
    return crop_img

# //////////////////////////////////////////////////////////////////////////////////////////



if __name__ == "__main__":
    test()
