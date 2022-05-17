#!/usr/bin/env python
# coding: utf-8

# ### 데이터 전처리

# In[4]:


from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt


# In[5]:


#seed값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


# In[6]:


#MNIST 데이터셋 불러오기
(X_train, Y_class_train),(X_test, Y_class_test)  = mnist.load_data()

print("학습셋 이미지 수: %d 개" %(X_train.shape[0]))
print("테스트셋 이미지 수: %d 개" %(X_test.shape[0]))


# In[7]:


#그래프로 확인
plt.imshow(X_train[0],cmap = "Greys")
#cmap="Greys"옵션을 지정해 흑백으로 출력
plt.show()


# reshape함수: 2차원배열 =>1차원 배열
#     
# np_utils.to_categorical(클래스,클래스개수): 원핫인코딩 적용

# In[9]:


#코드로 확인
for x in X_train[0]:
    for i in x:
        sys.stdout.write("%d\t" %i)
    sys.stdout.write("\n")


# In[10]:


#차원변환 과정
X_train = X_train.reshape(X_train.shape[0],28*28)
X_train = X_train.astype('float64')
X_train = X_train/255

X_test = X_test.reshape(X_test.shape[0],784).astype('float64')/255

#클래스값 확인
print("Class: %d" %(Y_class_train[0]))


# In[11]:


#바이너리화 과정
Y_train = np_utils.to_categorical(Y_class_train,10)
Y_test = np_utils.to_categorical(Y_class_test,10)

print(Y_test[0])


# ### 딥러닝 기본 프레임 

# In[12]:


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping

import os


# In[14]:


#seed값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


# In[15]:


#모델 프레임 설정

model = Sequential()
model.add(Dense(512,input_dim = 784,activation="relu"))
model.add(Dense(10,activation="softmax"))

#모델 실행 환경 설정
model.compile(loss = "categorical_crossentropy",optimizer = "adam",metrics=["accuracy"])


# In[17]:


#모델 최적화 설정

MODEL_DIR = "./model/"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
modelpath = "./model./{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath, monitor = "val_loss", verbose = 1,save_best_only = True)
early_stopping_callback = EarlyStopping(monitor="val_loss",patience=10)

#모델의 실행
history = model.fit(X_train,Y_train,validation_data = (X_test,Y_test),epochs = 30,batch_size = 200, verbose = 0,callbacks=[early_stopping_callback,checkpointer])

#테스트 정확도 출력
print("\n Test Accuracy: %.4f" %(model.evaluate(X_test,Y_test)[1]))

#테스트셋의 오차
y_vloss = history.history['val_loss']
#학습셋의 오차
y_loss = history.history['loss']

#그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len,y_vloss,marker=".",c ="red",label="Testset_loss")
plt.plot(x_len,y_loss,marker=".",c ="blue",label="Trainset_loss")

#그래프에 그리드를 주고 레이블을 표시
plt.legend(loc="upper right")
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[ ]:




