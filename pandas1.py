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

