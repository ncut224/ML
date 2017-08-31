import random
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#np.random.normal(mean,stdev,size)
'''我们随机生成 6 组 200 个的正态分布'''
x1 = np.random.normal(50, 6, 200)
y1 = np.random.normal(5, 0.5, 200)

x2 = np.random.normal(30,6,200)
y2 = np.random.normal(4,0.5,200)

x3 = np.random.normal(45,6,200)
y3 = np.random.normal(2.5, 0.5, 200)

'''x1、x2、x3 作为 x 坐标，y1、y2、y3 作为 y 坐标，两两配对。
(x1,y1) 标为 1 类，(x2, y2) 标为 2 类，(x3, y3)是 3 类。将它们画出得到下图，1 类是蓝色，2 类红色，3 类绿色。'''

plt.scatter(x1,y1,c='b',marker='s',s=50,alpha=0.8)
plt.scatter(x2,y2,c='r', marker='^', s=50, alpha=0.8)
plt.scatter(x3,y3, c='g', s=50, alpha=0.8)

