#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # Co-Training Regression

# In[2]:


def coreg(x_lbl,y_lbl,x_unlbl,K,T,P1,P2):
    num1,dim=np.shape(x_lbl)
    num2,_=np.shape(x_unlbl)
    if num1%2!=0:
        split=(num1+1)/2
    else:
        split=num1/2
    split=int(split)
    L1=x_lbl[0:split-1,:]
    Y1=y_lbl[0:split-1]
    L2=x_lbl[split:num1,:]
    Y2=y_lbl[split:num1]
    tmp=np.random.randint(2,num2,1)
    Up=x_unlbl[0:int(tmp),:]
    def nbh(xp,k,S):
        dist=np.linalg.norm(S-xp,axis=1)
        di=np.argsort(dist)
        di=di[0:k]
        return di
    def KNN_reg(x_train,y_train,x_test,K,P):
        num1,dim=np.shape(x_train)
        x_test=x_test.reshape(-1,dim)
        num2,_=np.shape(x_test)
        y_test=np.zeros(num2)
        #Step 1: K nearest neighbour
        Xi=np.repeat(x_test,num1,axis=0)
        Xj=np.tile(x_train,(num2,1))
        S=np.linalg.norm(Xi-Xj,ord=P,axis=1).reshape(num2,num1)
        S=np.argsort(S,axis=1)
        D=S[:,0:K-1]
        #Step 2: Prediction
        for i in range(num2):
            y_test[i]=np.mean(y_train[D[i,:]])
        return y_test
    for i in range(T):
        pi_x1=np.zeros((1,dim))
        pi_x2=np.zeros((1,dim))
        pi_y1=np.zeros(1)
        pi_y2=np.zeros(1)
        for j in range(2):
            err=[]
            for xu in Up:
                if j==0:
                    yu=KNN_reg(L1,Y1,xu,K,P1)
                    omega=nbh(xu,K,L1)
                    pt1=KNN_reg(L1,Y1,L1[omega,:],K,P1)
                    tL1=np.vstack((L1,xu.reshape(1,-1)))
                    tY1=np.hstack((Y1,yu))
                    pt2=KNN_reg(tL1,tY1,L1[omega,:],K,P1)
                    pt=Y1[omega]
                else:
                    yu=KNN_reg(L2,Y2,xu,K,P2)
                    omega=nbh(xu,K,L2)
                    pt1=KNN_reg(L2,Y2,L2[omega,:],K,P2)
                    tL2=np.vstack((L2,xu.reshape(1,-1)))
                    tY2=np.hstack((Y2,yu))
                    pt2=KNN_reg(tL2,tY2,L2[omega,:],K,P2)
                    pt=Y2[omega]
                err.append(np.sum((pt-pt1)**2-(pt-pt2)**2))
            if np.sum(np.array(err))!=0:
                ind=np.argmax(err)
                xj=Up[ind,:]
                if j==0:
                    yj=KNN_reg(L1,Y1,xj,K,P1)
                    pi_x1=np.vstack((pi_x1,xj.reshape(1,-1)))
                    pi_y1=np.hstack((pi_y1,yj))
                else:
                    yj=KNN_reg(L2,Y2,xj,K,P2)
                    pi_x2=np.vstack((pi_x2,xj.reshape(1,-1)))
                    pi_y2=np.hstack((pi_y2,yj))
                Up=np.delete(Up,ind,axis=0)
        pi_x1=np.delete(pi_x1,0,axis=0)
        pi_x2=np.delete(pi_x2,0,axis=0)
        pi_y1=np.delete(pi_y1,0,axis=0)
        pi_y2=np.delete(pi_y2,0,axis=0)
        L1=np.vstack((L1,pi_x2))
        Y1=np.hstack((Y1,pi_y2))
        L2=np.vstack((L2,pi_x1))
        Y2=np.hstack((Y2,pi_y1))
        tmp=int(np.random.randint(2,num2,1))
        Up=x_unlbl[0:tmp,:]
    p1=KNN_reg(L1,Y1,x_unlbl,K,P1)
    p2=KNN_reg(L2,Y2,x_unlbl,K,P2)
    p=np.hstack((p1.reshape(-1,1),p2.reshape(-1,1)))
    return(np.mean(p,axis=1))

