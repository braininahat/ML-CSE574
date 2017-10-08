import numpy as np
import pandas as pd

data=pd.read_csv('/Users/pranavsankhe/Desktop/CSE 574 ML/PROJECT 2/Querylevelnorm.txt', delimiter=' ',header=None);

temp=data.drop(data.columns[[0,1]],axis=1);
temp=temp.iloc[:,:46];


temp.iloc[:,:9] = temp.iloc[:,:9].applymap(lambda x: str(x)[2:]);
temp.iloc[:,9:] = temp.iloc[:,9:].applymap(lambda x: str(x)[3:]);

features=temp.as_matrix();

labels=data.iloc[:][1];
print(labels.shape);
print(features[:3][:]);

#partition the data
(r,c)=features.shape;
train_setX=features[:int(np.floor(0.8*r)),:]
val_setX=features[int(np.floor(0.8*r)):int(np.floor(0.9*r)),:]
test_setX=features[int(np.floor(0.9*r)):,:]

#size_test
train_setT = labels[:int(np.floor(0.8*r))]
val_setT = labels[int(np.floor(0.8*r)):int(np.floor(0.9*r))]
test_setT = labels[int(np.floor(0.9*r)):]


#weight matric creatiion and variable declaration
w=np.arange(r);







