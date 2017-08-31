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
fr=open("./disvurl1.txt","r")

ci=0;
while True:
    line=fr.readline()
    if line:
        ci=ci+1
        tokens = line.strip().split('|')
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
print("转换后yis:",y)


'''
import pandas as pd
pdx=pd.DataFrame(x)
print("framex is:",pdx)
print("type of framex is",pdx.dtypes)
print("dfhead: ",pdx.head(2))
print("dftail: ",pdx.tail(2))
'''