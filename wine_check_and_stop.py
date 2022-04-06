#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#모델 업데이트 함수와 학습 자동 중단 함수를 동시 사용


# In[6]:


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#seed값 설정
seed= 3
np.random.seed(seed)
tf.random.set_seed(seed)

#데이터 입력
df_pre = pd.read_csv("../dataset/wine.csv",header = None)
df = df_pre.sample(frac=0.15)
dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

#모델 설정
model = Sequential()
model.add(Dense(30,input_dim=12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

#모델 컴파일
model.compile(loss="binary_crossentropy",optimizer = "adam",metrics=['accuracy'])

#모델 저장 폴더 설정
MODEL_DIR ='./model/'               #모델을 저장하는 폴더
if not os.path.exists(MODEL_DIR):  #만일 위의 폴더가 존재하지 않으면
    os.mkdir(MODEL_DIR)            #이 이름의 폴더를 만들어 줌
    
#모델 저장 조건 설정
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'

#모델 업데이트 및 저장
#checkpointer라는 변수를 만들어 이곳에 모니터할 값을 지정, verbose=1이면 해당 함수의 진행 사항을 출력
checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True)

#학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',patience=100)

model.fit(X,Y,validation_split=0.2,epochs=3500,batch_size=500,verbose=0,callbacks=[early_stopping_callback,checkpointer])
    

#결과 출력
#print("\n Accuracy: %.4f" %(model.evaluate(X,Y)[1]))

