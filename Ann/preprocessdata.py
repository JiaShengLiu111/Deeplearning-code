# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # Solving the problem of Chinese character coding

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PrepDatas():
    """
    preprocess the data:StandardScaler+PCA
    """
    def __init__(self,X_train,X_test):
        self.X_train=X_train
        self.X_test=X_test
        self.X_train,self.X_test=self.preprocess(self.X_train,self.X_test) 
        
    def getDatas(self):
        return self.X_train,self.X_test
    
    def preprocess(self,X_train,X_test):
        """
        StandardScaler+PCA
        """
        # 数据归一化
        standardScaler = StandardScaler() 
        standardScaler.fit(X_train)
        X_train = standardScaler.transform(X_train)
        X_test = standardScaler.transform(X_test)
        
        # pca降维
        pca = PCA(0.95) # 只保留0.95的信息量
        pca.fit(X_train) 
        # 表示保留下来的主成分贡献率
        print("保留的主成分数目为："+repr(pca.n_components_))
        print("保留下来的主成分贡献率:"+str(pca.explained_variance_ratio_))  
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        # plt.plot([i for i in range(X_train.shape[1])],[np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
        # plt.show() 
        # column = repr(pca.n_components_)
        # print(X_reduction)
        print(X_train.shape)
        print(X_test.shape)
        return X_train,X_test
        