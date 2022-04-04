#!/usr/bin/env python
# coding: utf-8

# NAND 게이트와 OR게이트, 이 두 가지를 내재한 각각의 퍼셉트론이 다중 레이어안에서 작동하고, 이 두 가지 값에 대해 AND게이트를 수행한 값이 바로 Y_out값이다.
# 두 개의 노드를 둔 다층 퍼셉트론을 통해 XOR문제 해결가능

# In[10]:


import numpy as np

#가중치와 바이어스
w11 = np.array([-2,-2])
w12 = np.array([2,2])
w2 = np.array([1,1])

b1 = 3
b2 = -1
b3 = -1

#퍼셉트론
def MLP(x,w,b):
    y = np.sum(w*x) +b
    if y <= 0:
        return 0
    else:
        return 1
    
#NAND게이트
def NAND(x1,x2):
    return MLP(np.array([x1,x2]),w11,b1)

#OR게이트
def OR(x1,x2):
    return MLP(np.array([x1,x2]),w12,b2)

#AND게이트
def AND(x1,x2):
    return MLP(np.array([x1,x2]),w2,b3)

#XOR게이트
def XOR(x1,x2):
    return AND(NAND(x1,x2),OR(x1,x2))

#x1, x2값을 번갈아 대입하며 최종값 출력
if __name__ == "__main__":
    for x in [(0,0),(1,0),(0,1),(1,1)]:
        y = XOR(x[0],x[1])
        print("입력 값: "+str(x)+" 출력 값: "+str(y))

