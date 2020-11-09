import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
import numpy as np
import pywt
import os
import time
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from itertools import combinations
 
def snn_sim_matrix(data,k=50):
    """
    构造SNN相似度矩阵
    """
    samples_size, features_size = data.shape
    # 计算K近邻
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(data)
    # 记录每个样本的k个最近邻对应的索引
    knn_matrix = nbrs.kneighbors(data, return_distance=False)
    # snn相似度矩阵
    sim_matrix = np.zeros((samples_size, samples_size),dtype='int')  
    for i in range(samples_size):
        # 取出近邻为i的所有点索引
        t = np.where(knn_matrix == i)[0]
        # 对这些索引进行组合
        c = list(combinations(t, 2))
        for j in c:
            # K近邻稀疏操作
            if j[0] not in knn_matrix[j[1]]:
                continue
            else:
                # 两个点的共享近邻加一
                sim_matrix[j[0]][j[1]] += 1
                sim_matrix[j[1]][j[0]] += 1
    return sim_matrix


def snn_dens(snn_sim_matrix):
    """
    构造SNN图,计算SNN密度
    """
    snn_dens = np.zeros(snn_sim_matrix.shape[0],dtype='int')
    for i in range(len(snn_dens)):
        snn_dens[i] = np.sum(snn_sim_matrix[i])
        print(snn_dens[i])
    return snn_dens


def cluster(sim=None,den=None,corepts=100,minpts=10,eps=10):
    """
    基于密度和相似度矩阵进行聚类
    """
    # 选取核心点和噪声点
    core_idx = []
    noise_idx = []
    for i in range(len(den)):
        if den[i]>corepts:
            core_idx.append(i)
        if den[i]<minpts:
            noise_idx.append(i)
    core_idx=np.array(core_idx,dtype='int')
    noise_idx=np.array(noise_idx,dtype='int') 
    # 聚类结果，-1表示噪声点，0表示未分类，其余表示类别
    c = np.zeros(len(d),dtype='int')
    c_num=1
    q= queue.Queue(maxsize=0)
    visit = np.zeros(len(den),dtype='bool')
    # 将所有核心点置为可访问
    for i in core_idx:
        visit[i]=True
    # 除去噪声点
    for i in noise_idx:
        c[i]=-1
    # 取出尚未访问的核心点
    visitable=np.where(visit==True)[0]
    while len(visitable)!=0:
        # 从未访问的核心点中随机选择一个
        i = random.choice(visitable)
        q.put(i)
        print("first:",i)
        # 当前簇队列还有点未扫描
        while q.empty()==False:
            center=q.get()
            c[center]=c_num
            # 将点置为不可访问
            visit[i]=False
            # 取出尚未分类的点
            n_c=np.where(c==0)[0]
            for other in n_c:
                if sim[center][other]>eps:
                    c[other]=c_num
                    # 如果连接点是核心点，将点置为不可访问
                    if other in core_idx:
                        visit[other]=False
                        print("get core:",other)
                    q.put(other)
                    print("get:",other)
        c_num+=1
        # 取出可访问的核心点
        visitable=np.where((visit==True))[0]
    return c,core_idx


def main():
    x,y = make_blobs(n_samples=3000,n_features = 2,centers=[[300,400],[400,300],[400,300],[400,400]],
                cluster_std=[20,20,10,20],random_state=9)
    plt.scatter(x[:,0],x[:,1],s=10.,c='b')
    sim =snn_sim_matrix(x,k=35)
    d = snn_dens(sim)
    c,core = cluster(sim,d,corepts=1000,minpts=350,eps=25)
    plt.grid()
    plt.scatter(x[:,0],x[:,1],s=5.,c='b')
    plt.scatter(x[core,0],x[core,1],s=5.,c='r')
    plt.show()
    plt.scatter(x[:,0],x[:,1],s=5.,cmap='jet',c=clus)
    plt.show()
  
  if '__name__'=='__main__':
      main()





