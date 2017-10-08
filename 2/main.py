
import numpy as np
#Q1
fileX=open('/Users/pranavsankhe/Desktop/CSE 574 ML/PROJECT 2/Querylevelnorm_X.csv','rt')
fileT=open('/Users/pranavsankhe/Desktop/CSE 574 ML/PROJECT 2/Querylevelnorm_t.csv','rt')
features=np.loadtxt(fileX,delimiter=",")
labels=np.loadtxt(fileT,delimiter=",")

#Q2
(r,c)=features.shape
train_setX=features[:int(np.floor(0.8*r)),:]
val_setX=features[int(np.floor(0.8*r)):int(np.floor(0.9*r)),:]
test_setX=features[int(np.floor(0.9*r)):,:]

train_setT = labels[:int(np.floor(0.8*r))]
val_setT = labels[int(np.floor(0.8*r)):int(np.floor(0.9*r))]
test_setT = labels[int(np.floor(0.9*r)):]

#just a check print for the size of the matrices created
print(train_setX.shape,val_setX.shape,test_setX.shape)
print(train_setT.shape,val_setT.shape,test_setT.shape)

#Q3
# Closed form solution






