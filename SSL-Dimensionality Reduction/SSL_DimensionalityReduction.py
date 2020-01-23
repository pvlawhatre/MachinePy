#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np


# # SSDR-M, SSDR-CM and SSDR-CMU

# In[276]:


def SSDR_constrained(X,K,**kwargs):
    num,dim=np.shape(X)
    if len(kwargs)==0:
        kwargs['method']='M'
        kwargs['beta']=10
    #determining points in must-link and cannot-link
    try:
        beta=kwargs['beta']
        must_lnk=kwargs['ml']
        ml=np.ones(num)
        for i in range(num):
            ml[i]=np.sum(np.sum(np.isin(must_lnk,X[i,:]),axis=1)==dim)
        nm=np.sum(ml)
        alpha=kwargs['alpha']
        cannot_lnk=kwargs['cl']
        cl=np.ones(num)
        for i in range(num):
            cl[i]=np.sum(np.sum(np.isin(cannot_lnk,X[i,:]),axis=1)==dim)
        nc=np.sum(cl)
    except:
        beta=kwargs['beta']
        must_lnk=kwargs['ml']
        ml=np.ones(num)
        for i in range(num):
            ml[i]=np.sum(np.sum(np.isin(must_lnk,X[i,:]),axis=1)==dim)
        nm=np.sum(ml)
    else:
        pass
    #S matrix
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    if kwargs['method']=='M':
        mli=np.repeat(ml.reshape(-1,1),num,axis=0)
        mlj=np.tile(ml.reshape(-1,1),(num,1))
        ml=(mli+mlj).flatten()
        s=np.zeros(num*num)
        s[ml==2]=-beta/nm
        S=s.reshape((num,num))
    elif kwargs['method']=='CM':
        mli=np.repeat(ml.reshape(-1,1),num,axis=0)
        mlj=np.tile(ml.reshape(-1,1),(num,1))
        ml=(mli+mlj).flatten()
        cli=np.repeat(cl.reshape(-1,1),num,axis=0)
        clj=np.tile(cl.reshape(-1,1),(num,1))
        cl=(cli+clj).flatten()
        s=np.zeros(num*num)
        s[ml==2]=-beta/nm
        s[cl==2]=-alpha/nc
        S=s.reshape((num,num))
    else:
        mli=np.repeat(ml.reshape(-1,1),num,axis=0)
        mlj=np.tile(ml.reshape(-1,1),(num,1))
        ml=(mli+mlj).flatten()
        cli=np.repeat(cl.reshape(-1,1),num,axis=0)
        clj=np.tile(cl.reshape(-1,1),(num,1))
        cl=(cli+clj).flatten()
        s=np.ones(num*num)*(1/num**2)
        s[ml==2]=(-beta/nm)+(1/num**2)
        s[cl==2]=(-alpha/nc)+(1/num**2)
        S=s.reshape((num,num))
    #D matrix
    D=np.sum(S,axis=1)
    #Laplacian matrix
    L=D-S
    #A matrix
    A=np.dot(np.dot(X.T,L),X)
    #EVD 
    val,vec=np.linalg.eig(A)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #projection matrix
    p=vec[:,0:K]
    #Trnsformation
    y=np.dot(X,p)
    return (y)


# # SSDR-manifold

# In[ ]:


def SSDR_manifold(Xl,Yl,Xu,K,gamma):
    num_l,dim=np.shape(Xl)
    num_u,dim=np.shape(Xu)
    num=num_l+num_u
    X=np.vstack((Xl,Xu))
    def gauss(xi,xj,si,sj):
        return np.exp(-np.linalg.norm(xi-xj,axis=1)**2/(si*sj))
    #Step 1: labelled cost function
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    A=np.linalg.norm(Xi-Xj,axis=1).reshape((num,num))
    iA=np.argsort(A,axis=1)
    knn=iA[:,1:K+1]
    C_l=np.zeros((num,num))
    for i in range(num_l):
        #points in the same cluster
        C_l[i,knn[i,:]]=1
        ind=(Yl[i]!=Yl)
        #points in differnet cluster
        multiplier=ind*(-1)
        C_l[i,0:num_l]=C_l[i,0:num_l]*multiplier
        #unlabbeled points
        ind2=(knn[i,:]>=num_l)
        C_l[i,knn[i,:]]=0
    #Step 2: unlebelled cost function
    sigma=np.zeros(num)
    iA=np.sort(A,axis=1)
    sigma=iA[:,K]
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    Si=np.repeat(sigma,num,axis=0)
    Sj=np.tile(sigma,(num,1)).flatten()
    C_u=gauss(Xi,Xj,Si,Sj).reshape((num,num))
    C=C_l+gamma*C_u
    #Step 3: Laplacian of the graph
    D=np.diag(np.sum(C,axis=1))
    L=D-C
    T=np.dot(np.dot(X.T,L),X)
    #EVD(generalised eign value problem)
    val,vec=np.linalg.eig(T)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #projection matrix
    p=vec[:,0:K]
    #Trnsformation
    y=np.dot(X,p)
    return y

