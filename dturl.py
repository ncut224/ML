# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import pandas as pd

''' 数据读入 '''
''' 列表类型data'''
data = []
labels = []
fr=open("./disvurl1.txt","r")

ci=0;
while True:
    line=fr.readline()
    if line:
        ci=ci+1
        tokens = line.strip().split(',')
        data.append([str(tk) for tk in tokens[:-1]])
        #print("the",ci,"line is ",data)
        labels.append(tokens[-1])
    else:
        break
fr.close();

print("the",ci,"line is ",data)

x = np.array(data)
print("original y is :",np.array(labels))
#print("xis:",x)
labels = np.array(labels)
#print("labelsis:",labels)
#print("labels shape",labels.shape)

'''数据标准化'''
'''y标准化'''
y = np.zeros(labels.shape)
#print("yis:",y)
''' 标签转换为0/1/2/3/4/.. '''
y[labels=='BRAN']=0
y[labels=='XXXX']=1
y[labels=='SHOW']=2
y[labels=='EVAL']=3
y[labels=='PROL']=4
y[labels=='SUBS']=5
y[labels=='PASS']=6
pdy=pd.DataFrame(y)
print("转换后yis:",pdy)

'''x标准化'''


pdx=pd.DataFrame(x)
num_pdx=pdx.apply(lambda i: pd.factorize(i)[0]) # pd.factorize即可将分类变量转换为数值


#print("原来X数据集:\n",pdx.head(10))
#print("数值化X数据集:\n", num_pdx.head(10))


''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(num_pdx, y, test_size = 0.8)

''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)

#必须要将输入集转为 float 型
clf.fit(x_train, y_train)


''' 把决策树结构写入文件 '''
with open("./dt_arch_disvurl1.dot",'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''测试结果的打印'''
answer = clf.predict(num_pdx)
print("训练数据集:\n",num_pdx)
print("预测结果:\n",answer)
print("原结果:\n",y)
'''平均值,判断是否准确.'''
print(np.mean( answer == y))

