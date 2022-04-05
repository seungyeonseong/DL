#!/usr/bin/env python
# coding: utf-8

# 아이리스 품종 예측_다중 분류 문제

# In[1]:


#상관도 그래프
import pandas as pd
df = pd.read_csv("../dataset/iris.csv", names=["sepal_length","sepal_width","petal_length","petal_width","species"])
print(df.head())


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df,hue="species")
plt.show()


# In[4]:


#데이터 분류
dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]


# In[8]:


#문자열->숫자 형태
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)


# In[15]:


#Y값이 0과 1로 이루어져있어야함
from tensorflow.keras.utils import to_categorical

Y_encoded = to_categorical(Y)


# In[19]:


#모델 설정
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16,input_dim=4,activation="relu"))
model.add(Dense(3,activation ="softmax"))


# In[20]:


#모델 컴파일
model.compile(loss="categorical_crossentropy",optimizer ="adam",metrics=['accuracy'])

#모델 실행
model.fit(X,Y_encoded,epochs=50,batch_size=1)

#결과 출력
print("\n Accuracy: %.4f" %(model.evaluate(X,Y_encoded)[1]))

