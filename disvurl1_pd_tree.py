# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib

''' 数据读入 '''
names = ("hostvurl_ad,vurl1,vurl2,cate").split(',')

data = pd.read_csv('~/Downloads/mycode/ML/disvurl1.txt',names = names,delimiter=',',encoding="utf-8-sig")
print("原始数据为:\n",data.head(2))

'''复制给df'''
df=data;

'''列名,特证名'''
##column_names = df.columns.tolist();


'''根据特征值,进行数据的筛选'''
d1=df[(df.hostvurl_ad=='gq')&(df.vurl1=='brand')]
d2=df[df.vurl1.str.startswith('bran')]

'''d1 的记录数'''
len(d1)
len(d2)
'''d1数据统计探索-按cate分组统计'''
d1.groupby('cate').size()

'''对vurl1列进行索引'''
d3=d2.set_index(['vurl1'])

'''取vurl1='brand'的行'''
d3.loc['brand']

'''返回d3的第1行'''
d3.iloc[0]

'''取几行'''
df[:2]
df[2:5]
df['cate'].head(2)
df['cate'].tail(2)

'''获取vurl1列'''
df['vurl1'].head(2)

'''获取前三列x'''
df.iloc[:,0:3].head(5)
'''获取最后一列y'''
df.iloc[:,-1].head(5)

featuresNameList=['hostvurl_ad','vurl1','vurl2']
X=df[featuresNameList]

#另一种方法
#X=df.iloc[:,0:3] ##DataFrame
#第三种方法
##X=df[["hostvurl_ad","vurl1","vurl2"]]
features = list(X.columns[:4])
y1=df['cate']
#另一种方法
#y1=df.iloc[:,-1] ##Series
y=pd.DataFrame(y1) ##DataFrame

'''标准化X'''
#num_X=X.apply(lambda i: pd.factorize(i)[0]) # pd.factorize即可将分类变量转换为数值


'''自定义数字化函数'''
def encode_target(df, target_column):
    '''Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    '''
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

'''hostvurl_ad 列进行数字化'''
X2, targets = encode_target(X,"hostvurl_ad")

print("* X2.head()", X2[["Target", "hostvurl_ad"]].head(),
      sep="\n", end="\n\n")
print("* X2.tail()", X2[["Target", "hostvurl_ad"]].tail(),
      sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")

#print(X2.head(4))
'''重命名 新列'''
X2=X2.rename(columns = {'Target':'num_hostvurl_ad'})
#print(X2.head(4))

'''vurl1 列进行数字化'''
X2, targets = encode_target(X2,"vurl1")
X2=X2.rename(columns = {'Target':'num_vurl1'})


'''vurl2 列进行数字化'''
X2, targets = encode_target(X2,"vurl2")
X2=X2.rename(columns = {'Target':'num_vurl2'})

print("数字化X后的新DataFrame X2 为:\n",X2.head(4))

'''得到新的数字化X3'''
newfeatures=["num_hostvurl_ad","num_vurl1","num_vurl2"]
X3=X2[newfeatures]
print("新的数字化X3 为:\n",X3.head(5))

'''标准化y'''

'''hostvurl_ad 列进行数字化'''
y2, targets = encode_target(y,"cate")
y2=y2.rename(columns = {'Target':'num_cate'})

y3=y2["num_cate"]
print("数字话后的y3 is :\n",y3.head(5))


''' 拆分训练数据与测试数据  as DataFrame'''
x_train, x_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.8)

''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)

'''transfer DataFrame to Num Array'''
#x_train_array=x_train.values;
#y_train_array=y_train.values;

#x_train_array=pd.DataFrame.as_matrix(x_train)
#y_train_array=pd.DataFrame.as_matrix(y_train)


#必须要将输入集转为 float 型
model=clf.fit(x_train, y_train)
#clf.fit(x_train_array, y_train_array)

'''模型持久化'''
joblib.dump(clf, 'filename.pkl')
'''从磁盘加载模型'''
clf_pre = joblib.load('filename.pkl')


''' 把决策树结构写入文件 '''
with open("./dt_arch_disvurl1_pd.dot",'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''测试结果的打印'''
answer = clf_pre.predict(x_test)
print("训练数据集:\n",x_test.head(5))
print("预测结果:\n",answer)
print("原结果:\n",y_test)
'''平均值,判断是否准确.'''
print(np.mean( answer == y_test))

print(model_selection.cross_val_score(model, x_test, y_test, scoring='accuracy'))

#cross_val_score(model, x_test, y_test, scoring='wrong_choice')

def visualize_tree(tree_p, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        tree.export_graphviz(tree_p, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        tree.subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")





visualize_tree(clf, features)