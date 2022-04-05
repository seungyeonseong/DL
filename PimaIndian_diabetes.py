#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv("../dataset/pima-indians-diabetes.csv", names=["pregnant","plasma","pressure","thickness","insulin","BMI","pedigree","age", "class"])
print(df.head(5))


# In[2]:


print(df.info())


# In[3]:


print(df.describe())


# In[5]:


#as_index = False는 pregnant 정보 옆에 새로운 인덱스 만듦.
print(df[['pregnant','class']].groupby(['pregnant'], as_index = False).mean().sort_values(by='pregnant',ascending=True))


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), linewidths=0.1, vmax = 0.5, cmap=plt.cm.gist_heat, linecolor="white",annot=True)
plt.show()


# In[12]:


grid = sns.FacetGrid(df,col="class")
grid.map(plt.hist,'plasma',bins=10)
plt.show()


# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

#seed값 생성
np.random.seed(3)
tf.random.set_seed(3)

#데이터로드
dataset = np.loadtxt("../dataset/pima-indians-diabetes.csv",delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

#모델의 설정
model = Sequential()
model.add(Dense(12,input_dim=8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

#모델 컴파일
model.compile(loss="binary_crossentropy", optimizer = "adam",metrics=['accuracy'])

#모델 실행
model.fit(X,Y,epochs=200, batch_size =10)

#결과 출력
print("\n Accuracy: %.4f" %(model.evaluate(X,Y)[1]))

