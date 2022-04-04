#!/usr/bin/env python
# coding: utf-8

# 1.선형 회귀 실습

# (1)최소제곱법 method of least squares
# 주어진 독립변수의 값이 하나일 때 "최소제곱법" 적용
# 여러개의 x값이 주어지는 경우 "경사하강법" 적용

# In[2]:


import numpy as np

#x값과 y값
x = [2,4,6,8]
y = [81,93,91,97]

#x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:",mx)
print("y의 평균값:",my)

#기울기 공식의 분모
divisor = sum([(mx-i)**2 for i in x])

#기울기 공식의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i]-mx)*(y[i]-my)
    return d
dividend = top(x,mx,y,my)

print("분모:",divisor)
print("분자:",dividend)

#기울기와 y절편 구하기
a = dividend/divisor
b = my-(mx*a)

#출력으로 확인
print("기울기 a =",a)
print("y 절편 b =",b)



# (2)평균제곱오차 MSE

# In[4]:


import numpy as np

#기울기a와 y절편b
fake_a_b = [3,76]

#x,y의 데이터 값
data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#y = ax+b에 a와 b값을 대입하여 결과를 출력하는 함수
def predict(x):
    return fake_a_b[0]*x+fake_a_b[1]

#MSE 함수
def mse(y,y_hat):
    return ((y-y_hat)**2).mean()

#MSE함수를 각 y값에 대입하여 최종 값을 구하는 함수
def mse_val(y,predict_result):
    return mse(np.array(y),np.array(predict_result))

#예측값이 들어갈 빈 리스트
predict_result=[]

#모든 x값을 한 번씩 대입하여
for i in range(len(x)):
    #predict_result 리스트 완성
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제 점수=%.f, 예측 점수=%.f" %(x[i],y[i],predict(x[i])))
    
#최종 MSE출력
print("MSE 최종값: "+str(mse_val(predict_result,y)))


# (3)경사하강법 gradient descent 

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#공부 시간 x와 성적y의 리스트 만들기
data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#그래프로 나타내기
plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.show()

#리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기/인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함
x_data = np.array(x)
y_data = np.array(y)

#기울기 a와 절편 b의 값 초기화
a = 0
b = 0

#학습률 정하기
lr = 0.03

#몇 번 반복될지 결정
epochs = 2001

#경사하강법 시작
for i in range(epochs):
    y_pred = a*x_data + b    #에포크 수만큼 반복
    error = y_data - y_pred  #y를 구하는 식 세우기
    #오차 함수를 a로 미분한 값
    a_diff = -(2/len(x_data))*sum(x_data*error)
    #오차 함수를 b로 미분한 값
    b_diff = -(2/len(x_data))*sum(error)
    
    a = a - lr*a_diff    #학습률을 곱해 기존의 a값 업데이트
    b = b - lr*b_diff    #학습률을 곱해 기존의 b값 업데이트
    
    if i % 100 == 0:    #100번 반복될 때마다 현재의 a값, b값 출력
        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" %(i,a, b))
        
#앞서 구한 기울기와 절편을 이용해 그래프 다시 그리기
y_pred = a*x_data + b
plt.scatter(x,y)
plt.plot([min(x_data),max(x_data)],[min(y_pred),max(y_pred)])
plt.show()


# 2. 다중 선형 회귀 multi linear regression

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  #3D 그래프 그리는 라이브러리

#공부 시간 x와 성적 y의 리스트 만들기
data = [[2,0,81],[4,4,93],[6,2,91],[8,3,97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

#그래프로 확인
ax = plt.axes(projection="3d")
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist =11
ax.scatter(x1,x2,y)
plt.show()

#리스트로 되어있는 x와 y값을 넘파이 배열로 바꾸기/인덱스로 하나씩 불러와 계산하기 위해
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

#기울기 a와 절편 b의 초기화
a1,a2,b = 0, 0, 0

#학습률
lr = 0.02

#몇 번 반복할지 설정(0부터 세므로 원하는 반복 횟수에 +1)
epochs = 2001

#경사하강법 시작
for i in range(epochs):    #epoch 수 만큼 반복
    y_pred = a1*x1_data + a2*x2_data + b    #y를 구하는 식 세우기
    error = y_data - y_pred    #오차를 구하는 식
    #오차함수를 a1로 미분한 값
    a1_diff = -(2/len(x1_data))*sum(x1_data*error)
    #오차함수를 a2로 미분한 값
    a2_diff = -(2/len(x2_data))*sum(x2_data*error)
    #오차 함수를 b로 미분한 값
    b_diff = -(2/len(x1_data))*sum(error)
    
    a1 = a1 - lr*a1_diff    #학습률을 곱해 기존의 a1값 업데이트
    a2 = a2 - lr*a2_diff    #학습률을 곱해 기존의 a2값 업데이트
    b = b - lr*b_diff       #학습률을 곱해 기존의 b값 업데이트
    
    if i %100==0:    #100번 반복될 때마다 현재의 a1, a2, b값 출력
        print("epoch=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" %(i, a1, a2, b))
    
    


# In[ ]:




