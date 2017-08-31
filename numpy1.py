import numpy as np

A=np.arange(12).reshape(3,4)

print(A)


np.mean(A,axis=0)
np.sum(A)
#np.累加(a)
#A[3,1]
#A[3,:]
#A[:,1]
#A.shape
C=np.concatenate((A,A,A),axis=0)
print(C)
#np.vstack((A,A,A))
C=np.concatenate((A,A,A),axis=1)
print(C)
#np.hstack((A,A,A))

print(A)
#axis=1 对列进行操作
#分割
print(np.split(A,2,axis=1))

#不等分割
print(np.array_split(A,3,axis=1))

#纵向等分
print(np.vsplit(A,3))
#横向等分
print(np.hsplit(A,2))


print("新的一章\n")
a=np.arange(4)
print(a)
b = a
c = a
d = b
e = a.copy()
a[0]=11
print(a)

print(b is a)
print(c)
print(e is a)


d[1:3]=[22,33]
print(d)

