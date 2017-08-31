# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
s = pd.Series([1,3,6,np.nan,44,1])
print(s)
print(type(s))

dates = pd.date_range('20160101',periods=6)
print(dates)

df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print(df)
print(type(df))

df1=pd.DataFrame(np.arange(12).reshape(3,4),columns=['a','b','c','d'])
print(df1)

df2=pd.DataFrame(
    {'A': 1.,
     'B': pd.Timestamp('20130102'),
     'C': pd.Series(1,index=list(range(4)),dtype=float),
     'D': np.array([3]*4,dtype='int32'),
     'E': pd.Categorical(["test","train","test","train"]),
     'F': 'foo'
    }
)

print(df2)

print(df2.dtypes)
print(df2.index)
print(df2.columns)
print(df2.values)
print(df2.describe())

#print(df2.T)

print(df2.sort_index(axis=1,ascending=False))

print(df2.sort_index(axis=0,ascending=False))

print(df2.sort_values(by='E'))

'''如何进行选择数据'''


dates = pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.arange(24).reshape(6,4),index=dates,columns=['A','B','C','D'])

print(df)

print(df['A'],df.A)

print(df[0:3],df['20130102':'20130104'])

#select by label:loc
'''行选择'''
print(df.loc['20130102'])
'''列选择'''
print(df.loc[:,['A','B']])



print(df.loc['20130102',['A','B']])


#select by position:iloc
'''第三行'''
print(df.iloc[3])

print(df.iloc[3,1])

print(df.iloc[3:5,1:3])

print(df.iloc[[1,3,5],1:3])


#mixed selection:ix
print(df.ix[:3,['A','C']])


#Boolean indexing
print(df)
print(df[df.A > 8])


'''设置值'''

df.iloc[2,2] =1111
print(df)

df.loc['20130101','B'] = 2222

print(df)


#df[df.A > 4] = 0
#print(df)

df.A[df.A > 4] = 0
print(df)

'''添加列'''
df['F'] =  np.nan
print(df)

df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101', periods=6))
print(df)

'''填充缺失值'''
dates = pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.arange(24).reshape(6,4),index=dates,columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)


'''按行 丢掉non的记录'''
#print(df.dropna(axis=0,how='any'))  #how={'any', 'all'}


'''填为0'''
#print(df.fillna(value=0))

print(df.isnull())

'''至少有一个是nan的值'''
print(np.any(df.isnull()) == True)


'''pandas 的数据导入导出'''

#read_csv,read_excel,read_hdf,read_sql,read_json,read_msgpack,read_html,read_gbq,read_stata,read_sas,read_clipboard,read_pickle
#to_csv,to_excel,to_hdf,to_sql,to_json,to_msgpack,to_html,to_gbq,to_stata,to_sas,to_clipboard,to_pickle


#drugs = pd.read_csv('/Users/admin/Downloads/DownloadRoutine/drugs.csv')

drugs = pd.read_excel('/Users/admin/Downloads/DownloadRoutine/drugs2.xlsx')
#print(drugs)

drugs.to_pickle('/Users/admin/Downloads/DownloadRoutine/drugs2.pickle')


'''合并 concat'''

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

print(df1)

#verical concat 纵向
res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
print(res)


'''join,['inner','outer']'''
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

print(df1)
print(df2)

'''默认为outer'''
res1 = pd.concat([df1,df2])
print(res1)

'''改为inner'''
res1 = pd.concat([df1,df2],join='inner', ignore_index=True)
print(res1)


'''join_axes'''
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
res = pd.concat([df1,df2], axis=1, join_axes=[df1.index])

print(res)

'''append'''
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'])
df3 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'])
res = df1.append(df2, ignore_index=True)
print(res)
res = df1.append([df2,df3], ignore_index=True)
print(res)


'''给dataframe 添加 series  一行'''
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res = df1.append(s1, ignore_index=True)
print(res)

'''合并merge'''
# merging two df by key/keys. (may be used in database)
# simple example
left =  pd.DataFrame({'key': ['K0','K1','K2','K3'],
                      'A': ['A0','A1','A2','A3'],
                      'B': ['B0','B1','B2','B3']})

right = pd.DataFrame({'key': ['K0','K1','K2','K3'],
                      'C': ['C1','C2','C3','C4'],
                      'D': ['D1','D2','D3','D4']})

print(left)
print(right)

res = pd.merge(left, right, on='key')
print(res)

# merging two df by key/keys. (may be used in database)
# consider two keys
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0','C1', 'C2', 'C3'],
                      'D': ['D0','D1', 'D2', 'D3']})

print(left)
print(right)

# inner join
res = pd.merge(left, right, on=['key1','key2'],how='inner')  # how=['left','right','outer','inner']
print(res)

# inner join
res = pd.merge(left, right, on=['key1','key2'],how='outer')
print(res)


# indicator
df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print(df1)
print(df2)
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print(res)
# give the indicator a custom name
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
print(res)


# merged by index
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                                  'B': ['B0', 'B1', 'B2']},
                                  index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                                     'D': ['D0', 'D2', 'D3']},
                                      index=['K0', 'K2', 'K3'])
print(left)
print(right)
# left_index and right_index
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
print(res)
res = pd.merge(left, right, left_index=True, right_index=True, how='inner')
print(res)


# handle overlapping
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
print(boys)
print(girls)
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res)

# join function in pandas is similar with merge. If know merge, you will understand join

'''plot 画图'''
#from __future__ import print_function
#import matplotlib.pyplot as plt

# Series
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
#data.plot()
#plt.show()


#DataFrame
data =  pd.DataFrame(np.random.randn(1000,4), index=np.arange(1000), columns=list("ABCD"))

data = data.cumsum()


#plot methods:
# 'bars', 'hist', 'box', 'kde', 'area', 'scatter','hexbin', 'pie'
print(data.head())
#data.plot()
ax = data.plot.scatter(x='A',y='B', color = 'DarkBlue', label='Class 1')
data.plot.scatter(x='A',y='C', color='DarkGreen', label='Class 2', ax=ax)
plt.show()