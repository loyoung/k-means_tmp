"""
    模式识别作业
    K-means聚类算法
"""
__author__ = 'yangl'

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import csv
import folium

# 从文件读取数据
def fileLoader(filename):
    data = []
    with open(filename) as f:
        next(f)
        for line in f.readlines():
            t_data = line.strip().split(',')
            # data.append(t_data)
            # 经度，纬度，震级，深度
            data.append([t_data[3],t_data[2],t_data[1],t_data[4]])
    # data = pd.read_csv("earthquake1.csv")
    return data

# 两点的欧氏距离
def distance(e1, e2):
    dis = np.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2)
    return dis

# 随机获取k个初始聚类中心
def createCent(dataSet, k):
    center = random.sample(list(dataSet), k)
    return np.array(center)

# arr中距离 p 最近的元素，用于聚类
def closest(p, arr):
    c = 0
    min_d = distance(p, arr[0])
    for index,e in enumerate(arr):
        d = distance(p, e)
        if d < min_d:
            min_d = d
            c = index
    return min_d,c

# 保存数据
def save_csv(filename,data):
    f = open(filename, 'w')
    writer = csv.writer(f)
    for i in data:
        writer.writerow(i)
    f.close()

def kMeans(dataSet, k):
    # 获取数据有多少条
    numSamples = np.shape(dataSet)[0]
    # 初始化所有点的簇和与簇中心的距离
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    # 创建 k 个初始聚类中心
    clusterCenter = createCent(dataSet, k)
    # clusterCenter = np.array(random.sample(list(dataSet), k))
    # sameCluster = []
    changeFlag = True
    n = 0  # 迭代次数
    while (changeFlag & (n<2000)):
        print("第 %d 次迭代....."%(n))
        n += 1
        changeFlag = False
        # 给每个点指派簇

        for i in range(numSamples):
            print("第 %d 次迭代,计算第 %d 个点....."%(n,i))
            # 离 i 最近的簇中心
            minDist,minIndex = closest(dataSet[i],clusterCenter)

            # minDist = distance(dataSet[i],clusterCenter[0])
            # minIndex = 0
            # for j in range(1,k):
            #     distIJ = distance(clusterCenter[j],dataSet[i])
            #     if distIJ < minDist:
            #         minDist = distIJ
            #         minIndex = j

            # 如果点的簇发生改变就repeat
            if clusterAssment[i, 0] != minIndex:
                changeFlag = True
                print("-------第 %d 次迭代,第%d个点发生了簇改变.....-----------"%(n, i))
            # 取第 i 行所有数据,更新点 i 的簇和与簇中心的距离,保留四位小数
            clusterAssment[i, :] = int(minIndex), round(minDist ** 2,4)

        # 重新计算聚类中心
        for cent in range(k):
            sameCluster = []  # 临时保存同一个簇的点
            # 矩阵.A是把矩阵转换为数组numpy
            clusterPoint_cent = np.nonzero(clusterAssment[:, 0].A == cent)[0] # 返回 clusterAssment 中簇值等于cent 的点的索引。
            # 簇值为 cent 的点
            for i in clusterPoint_cent:
                sameCluster.append(dataSet[i])
            # sameCluster = dataSet[clusterCenter_cent]
            # sameCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 求簇中心，保留六位小数
            clusterCenter[cent, :] = np.mean(sameCluster, axis=0)
            clusterCenter[cent,0] = round(clusterCenter[cent,0],4)
            clusterCenter[cent,1] = round(clusterCenter[cent,1],4)
    save_csv("clusterResult.csv",clusterAssment)
    save_csv("clusterCenter.csv", clusterCenter)
    return clusterCenter, clusterAssment.A


if __name__ == "__main__":
    # data = fileLoader('earthquake1.csv')
    # data = np.array(data)

    # 经度，纬度，震级，深度
    filedata = pd.read_csv("earthquake.csv")
    # filedata = pd.read_csv("earthquake1.csv")
    data = []
    print("-------------------- loading data..... ---------------")
    for i in range(len(filedata)):
        # earthquake.csv
        data.append([filedata.values[i][2],filedata.values[i][1],filedata.values[i][3],filedata.values[i][0]])
        # earthquake1.csv
        # data.append([filedata.values[i][3],filedata.values[i][2],filedata.values[i][1],filedata.values[i][4]])
    # print(len(data))
    print("-------------------- loading data done ---------------")
    print("-------------------- kmeans computing..... ---------------")
    # 聚类中心，聚类结果
    k = 3
    clusterCenter, clusterAssment = kMeans(data,k)
    print(clusterCenter)
    col = ['red', 'orange','yellow', 'blue', 'green', 'purple', 'cyan', 'black']
    marker = ['.', 'v', 's', '*', '+', 'x', '>', '^']
    print("-------------------- start plt ---------------")
    # plt.scatter(data[:,0], data[:,1], s=5*data[:,2],c=col[int(clusterAssment[i][0])])
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], s=3*data[i][2],c=col[int(clusterAssment[i][0])], marker=marker[int(clusterAssment[i][0])])
    for i in range(len(clusterCenter)):
        plt.scatter(clusterCenter[i][0],clusterCenter[i][1],s=40*clusterCenter[i][2],c=col[i], marker=marker[i])
    print("--------------------  plt done ---------------")
    plt.show()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("successed!")





