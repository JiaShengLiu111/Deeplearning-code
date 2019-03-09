# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # Solving the problem of Chinese character coding

import numpy as np

class Datas():
    """
    read datas from files and organize them into feature vectors
    """
    def __init__(self,negative,positive,testneg,testpos):
        """
        init the filepaths of datas
        """
        self.negative = negative
        self.positive = positive
        self.testneg = testneg
        self.testpos = testpos
        # the length of feature vectors equaling with 373
        self.feature_length = 373
        # show the information or not in reading the datas
        self.show = False
        # read datas
        self.X_train,self.y_train,self.X_test,self.y_test = self.readDatas() 
    
    def getDatas(self):
        return self.X_train,self.y_train,self.X_test,self.y_test
    
    def readLines4file(self,fileFullPath): 
        """
        按行读取txt文件内容，并存储于列表中
        """
        result = [] 
        f = open(fileFullPath) 
        line = f.readline() 
        # result.append(line)
        while line:
            # print line
            result.append(line)
            line = f.readline()
        f.close()
        return result   
    
    def getData(self,testContent): 
        """
        将testContent转化为特征向量：将文件内容按行相加...,然后按照空格划分...,最后提取特征向量
        """
        # 首先将文件内容按行相加
        value = ''
        for i in range(len(testContent)):
            value = value + testContent[i] 
        # 字符串替换：将value中“</data>”替换为“<data>”
        value = value.replace('</data>','<data>') 
        # 将value按照“<data>”划分
        value_split = value.split('<data>')
        if self.show:
            print("划分结果中一共包含"+str(len(value_split))+"段")
        data = value_split[1]
        if self.show:
            print("其中data段部分内容为：")
            print data[:500]

        # 下面对data按照空格划分
        results = data.split(' ')
        # 将results中每个元素中'\r\n替换为空'
        for i in range(len(results)):
            results[i]=results[i].replace('\r\n','') 

        # 将results中不包含数字的成分（空''）去除掉
        numbers = []
        for i in range(len(results)):
            if results[i]=='':
                pass
            else:
                numbers.append(results[i])
        # print numbers[:100]
        print "文件中数字总个数为："+str(len(numbers)) 

        # 将numbers中元素类型转化为数值型
        for i in range(len(numbers)):  
            numbers[i]=float(numbers[i])

        # 下面将所有数字组装成特征向量
        features = []
        tmp = []
        for i in range(len(numbers)): 
            if len(tmp)<self.feature_length:
                tmp.append(numbers[i]) 
            else:
                features.append(tmp)
                tmp = []
                tmp.append(numbers[i])
        features.append(tmp)
        return features  
    
    def getFeatures(self,fileFullPath):
        """
        输入一个文件路径fileFullPath，返回文件中的特征向量
        """
        testContent = self.readLines4file(fileFullPath) 
        features = self.getData(testContent)
        return np.array(features)
    
    def unionList(self,list1,list2):
        """
        将两个list取并
        """
        result = list(list1)
        for i in range(len(list(list2))):
            result.append(list2[i])
        return np.array(result)
    
    def readDatas(self):
        """
        根据输入的各文本文件的路径，读取正负样本，并组织成特征向量
        """
        features_neg = self.getFeatures(self.negative) 
        print "特征向量维度为："+str(features_neg.shape)+"\n"

        features_pos = self.getFeatures(self.positive) 
        print "特征向量维度为："+str(features_pos.shape)+"\n"

        features_testneg = self.getFeatures(self.testneg) 
        print "特征向量维度为："+str(features_testneg.shape)+"\n"

        features_testpos = self.getFeatures(self.testpos) 
        print "特征向量维度为："+str(features_testpos.shape)+"\n"
        
        # 构建训练集
        X_train = self.unionList(features_neg,features_pos)
        y_train = [[1,0]]*len(features_neg)+[[0,1]]*len(features_pos)
        y_train = np.array(y_train)
        print "训练集维度："+str(X_train.shape)
        print "训练集标签维度："+str(y_train.shape)

        # 构建测试集
        X_test = self.unionList(features_testneg,features_testpos)
        y_test = [[1,0]]*len(features_testneg)+[[0,1]]*len(features_testpos)
        y_test = np.array(y_test)
        print "测试集维度："+str(X_test.shape)
        print "测试集标签维度："+str(y_test.shape)
        return X_train,y_train,X_test,y_test
        
        