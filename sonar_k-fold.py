#!/usr/bin/env python
# coding: utf-8

# In[5]:


from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import tensorflow as tf

#seed값 설정
np.random.seed(0)
tf.random.set_seed(0)

#데이터 입력
df = pd.read_csv("../dataset/sonar.csv",header=None)

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

#문자열 변형
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

#10개의 파일로 쪼갬
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state = 0)

#빈 accuracy 설정
accuracy = []

#모델 설정, 컴파일, 실행
for train, test in skf.split(X,Y):
    model = Sequential()
    model.add(Dense(24,input_dim=60,activation="relu"))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics=['accuracy'])
    model.fit(X[train],Y[train],epochs=100,batch_size=5)
    k_accuracy = "%.4f" %(model.evaluate(X,Y)[1])
    accuracy.append(k_accuracy)
    
#결과 출력
print("\n %.f fold accuracy:" %n_fold, accuracy)

