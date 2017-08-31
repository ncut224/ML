# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split


''' 数据读入 '''
''' 列表类型data'''
data = []
labels = []
fr=open("./1.txt","r")

ci=0;
while True:
    line=fr.readline()
    if line:
        ci=ci+1
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        #print("the",ci,"line is ",data)
        labels.append(tokens[-1])
    else:
        break
fr.close();


x = np.array(data)
print("xis:",x)
labels = np.array(labels)
print("labelsis:",labels)
print("labels shape",labels.shape)
y = np.zeros(labels.shape)
print("yis:",y)
''' 标签转换为0/1 '''
y[labels=='fat']=1
print("转换后yis:",y)


''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

''' 把决策树结构写入文件 '''
with open("./dt_arch.dot",'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''测试结果的打印'''
answer = clf.predict(x_train)
print("训练数据集:\n",x_train)
print("预测结果:\n",answer)
print("原结果:\n",y_train)
'''平均值,判断是否准确'''
print(np.mean( answer == y_train))

