# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:34:05 2020

@author: Jindongwang; @modified by: Laplace

The MIT license applied
"""
#@from numba import cuda
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier # 2 ways for base learner
#from sklearn.tree import DecisionTreeRegressor

def kernel(ker, X1, X2, gamma):
    """
    sklearn.metrics.pairwise 模块实现了对距离和样本相似性的评估,包含了距离测度(distance metrics)和核函数(kernel)
    linear_kernel 计算线性核函数: k(x,y) = x.T * y
    rbf_kernel 计算rbf核
    """
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)

    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K
''' 
# Debug needed!  
    elif ker == 'rbf':
        if X2 is not None:
                sum_x2 = np.sum(np.multiply(X2, X2), axis=1)
                sum_x2 = sum_x2.reshape((len(sum_x2), 1))
                K = np.exp(-1 * (
                    np.tile(np.sum(np.multiply(x1, x1), axis=1).T, (n2, 1)) + np.tile(sum_x2, (1, n1)) - 2 * np.dot(X2,X1.T)) / (dim * 2 * 1))
        else:
                P = np.sum(np.multiply(X1, X1), axis=1)
                P = P.reshape((len(P), 1))
                K = np.exp(-1 * (np.tile(P.T, (n1, 1)) + np.tile(P, (1, n1)) - 2 * np.dot(x1, x1.T)) / (dim * 2 * 1))
    return K
'''

class TCA:
    """
    This is an algorithm for transfer component analysis based on Domain Adaptation via Transfer Component Analysis
    """
    def __init__(self, kernel_type='rbf', dim=10, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma  
        
   # @cuda.jit
    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T)) # 组合两组数据
        X = X / np.linalg.norm(X, axis=0) # 求解矩阵的范数 |axis=0表示按列向量处理，求多个列向量的范数|axis=None表示矩阵范数
        m, n = X.shape # m行, n列
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1)))) # 垂直（按照行顺序）的把数组给堆叠起来
        M = e * e.T # array的转置
        M = M / np.linalg.norm(M, 'fro') # 求解矩阵的范数，范数类型 = 'fro'
        H = np.eye(n) - 1 / n * np.ones((n, n)) # np.eye()生成对角矩阵, H: 中心矩阵
        K = kernel(self.kernel_type, X, None, gamma = self.gamma) # 针对X的核函数计算, MMD = 1/m^2 * K(X,X) + 1/n^2 * K(Y,Y) - 2/mn * K(X,Y)
        n_eye = m if self.kernel_type == 'primal' else n # 赋值对角矩阵维度
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T]) # np.linalg.multi_dot()矩阵乘法运算 
        '''
        这里解决的是迁移学习简明手册10.1中给出的精炼公式：(X * M * X.T + lambda * I) * A = X * H * X.T * A * phi
        '''
        w, V = scipy.linalg.eig(a, b) # w--特征值; v--特征向量
        ind = np.argsort(w)# 从小到大排序
        A = V[:, ind[:self.dim]] # 取ind排序中小的dim列(降维)
        Z = np.dot(A.T, K) # 矩阵乘法
        Z = Z/ np.linalg.norm(Z, axis=0) # 求解矩阵的范数|axis=0表示按列向量处理，求多个列向量的范数
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new    
    
    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        #clf = DecisionTreeRegressor()
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred   
        
        
    
    
'''
if __name__ == '__main__':
    domains = ['DDoS_Bot.mat', 'Allbot_2.mat', 'allbenbot.mat', 'Botwith2.mat']
    for i in [0]:
            for j in [1]:
                if i != j:
                    src, tar = 'data/' + domains[i], 'data/' + domains[j]
                    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                    Xs, Ys, Xt, Yt =  src_domain['fts'][:-2000:50],src_domain['labels'][:-2000:50], tar_domain['fts'], tar_domain['labels']
                    tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                    acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
                    print(acc)

'''
if __name__ == '__main__':
    domains = ['DDoS_Bot.mat', 'AllBot_2.mat', 'allddos.mat', 'Botwith2.mat']
    for i in [0]:
            for j in [1]:
                if i != j:
                    src, tar = 'data/' + domains[i], 'data/' + domains[j]
                    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                    Xs, Ys, Xt, Yt = np.append(src_domain['fts'][:-2000:50],src_domain['fts'][-1960:-1:19],axis = 0),np.append(src_domain['labels'][:-2000:50],src_domain['labels'][-1960:-1:19],axis = 0), tar_domain['fts'], tar_domain['labels']
                    tca = TCA(kernel_type='rbf', dim=5, lamb=1, gamma=1)
                    acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
                    print(acc)


'''
np.append(src_domain['fts'][:-2000:50],src_domain['fts'][-2000:-1:15],axis = 0)
np.append(src_domain['labels'][:-2000:50],src_domain['labels'][-2000:-1:15],axis = 0)
'''



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_label = Yt  # 非二进制需要pos_label
y_pre = ypre
fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve: 20 Bot instances added in training domain')
plt.legend(loc="lower right")
plt.show()


'''
acc with DT: 0.9701648783041089
acc with KNN: 0.9761842449620518
acc by using target domain to train and test with all BOT: 0.582909460834181
acc by using allBot as train and test with DDoS: 0.5667774086378737
acc by using DDoS as train and test with AllDDoS: 0.9937124111536358
acc by Bot train & AllDDoS test: 0.20226474033580633
acc by Bot train & allBot as test: 0.6439471007121058
acc by Bot train & DDoS test: 0.43145071982281286
acc by Bot train & Bot tset: 0.9950274797173515
'''

'''                    
accuracy of SVM with Source+Target datasets: 0.8586757393352525
accuracy of SVM with Source dataset: 0.9685946087411672
accuracy of SVM with Target dataset: 0.9248887725726249
'''
       
'''
# plot pca for predicted data vs original data
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA, IncrementalPCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import matplotlib.pyplot as plt
n_components = 3
X = Xt
Y = Yt
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
ax = plt.subplot(111, projection='3d')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 200
x, y, z = X_ipca[:,0], X_ipca[:,1], X_ipca[:,2]
times_0 = 0
times_1 = 0
times_2 = 0
times_err = 0
for i in range(len(X)):
    if Y[i] != y_pre[i]:
        times_err += 1
        if times_err == 1:            
            ax.scatter(x[i], y[i], z[i], c='black',s= 15,marker='*', label = 'Undetected')
        else:
            ax.scatter(x[i], y[i], z[i], c='black',s= 15,marker='*')
    else:          
        if Y[i] == 0:
            times_0 += 1
            if times_0 == 1:           
                ax.scatter(x[i], y[i], z[i], c='b',s= 5,marker='o', label = 'Normal')
            else:
                ax.scatter(x[i], y[i], z[i], c='b',s= 5,marker='o')
        elif Y[i] == 1:
            times_1 += 1
            if times_1 == 1:  
                #continue
                ax.scatter(x[i], y[i], z[i], c='r',s= 5,marker='>', label = 'DDoS')
            else:
                #continue
                ax.scatter(x[i], y[i], z[i], c='r',s= 5,marker='>')
        else:       
            times_2 += 1
            if times_2 ==1:
                #continue
                ax.scatter(x[i], y[i], z[i], c='r',s= 20,marker='s', label = 'Bot')
            else:
                #continue
                ax.scatter(x[i], y[i], z[i], c='r',s= 20,marker='s')
ax.view_init(40, -45)
plt.legend(fontsize=13)
plt.savefig('TCA_allbot_IPCA_results.png', dpi = 100)
plt.show()

'''

          
    