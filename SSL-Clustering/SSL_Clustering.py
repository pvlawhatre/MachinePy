#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # Constrained K-means

# In[77]:


def constrained_kmeans(x_lbl,y_lbl,x_unlbl):
    n1,d=np.shape(x_lbl)
    n2,_=np.shape(x_unlbl)
    y_unlbl=np.zeros(n2)
    K=int(np.size(x_lbl)/d)
    C=np.zeros((K,d))
    #Step 1: centre initialisation
    for i in range(K):
        tx=x_lbl[i==y_lbl,:]
        C[i,:]=np.mean(tx,axis=0)
    #Step 2:Points assignment
    y_tmp=np.zeros_like(y_unlbl)
    flag=0
    x_tot=np.vstack((x_lbl,x_unlbl))
    y_tot=np.hstack((y_lbl,y_unlbl))
    pdist=0
    while True:
        Xi=np.repeat(x_unlbl,K,axis=0)
        Xj=np.tile(C,(n2,1))
        S=np.linalg.norm(Xi-Xj,axis=1).reshape(n2,K)
        tmp=np.argsort(S,axis=1)
        y_unlbl=tmp[:,0]
        dist=np.linalg.norm(y_lbl)
        print(dist)
        if np.abs(dist-pdist)<=0.0001:
            flag+=1
        else:
            flag=0
        if flag==3:
            break
        pdist=dist
        y_tot[n1:(n1+n2)]=y_unlbl
        for i in range(K):
            tx=x_tot[i==np.array(y_tot),:]
            C[i,:]=np.mean(tx,axis=0)
        pdist=dist
    return(y_unlbl)


# # Seed K-means

# In[132]:


def seed_kmeans(x_lbl,y_lbl,x_unlbl):
    n1,d=np.shape(x_lbl)
    n2,_=np.shape(x_unlbl)
    y_unlbl=np.zeros(n2)
    K=int(np.size(x_lbl)/d)
    C=np.zeros((K,d))
    #Step 1: centre initialisation
    for i in range(K):
        tx=x_lbl[i==y_lbl,:]
        C[i,:]=np.mean(tx,axis=0)
    #Step 2:Points assignment
    y_tmp=np.zeros_like(y_unlbl)
    flag=0
    x_tot=np.vstack((x_lbl,x_unlbl))
    y_tot=np.hstack((y_lbl,y_unlbl))
    pdist=0
    X=np.vstack((x_lbl,x_unlbl))
    Y=np.hstack((y_lbl,y_unlbl))
    while True:
        Xi=np.repeat(X,K,axis=0)
        Xj=np.tile(C,(n2+n1,1))
        S=np.linalg.norm(Xi-Xj,axis=1).reshape(n2+n1,K)
        tmp=np.argsort(S,axis=1)
        Y=tmp[:,0]
        dist=np.linalg.norm(Y)
        print(dist)
        if np.abs(dist-pdist)<=0.0001:
            flag+=1
        else:
            flag=0
        if flag==3:
            break
        pdist=dist
        for i in range(K):
            tx=X[i==Y,:]
            C[i,:]=np.mean(tx,axis=0)
        pdist=dist
    return(X,Y)


# # COP Kmeans

# In[ ]:


def cop_kmean(X,K,must_lnk,cannot_lnk):
    num,dim=np.shape(X)
    def kpp(X,K):
        pts,dim=np.shape(X)
        #1 st centre
        c=np.zeros((K,dim))
        tmp=np.random.randint(0,pts)
        c[0,:]=X[int(tmp),:]
        #centres drawn from distributioin of normalised distance from neareat centre
        D=np.ones((pts,K-1))
        D=D*np.inf
        for i in range(1,K):
            #distance matrix
            D[:,i-1]=(np.linalg.norm(X-c[i-1,:],axis=1))**2
            #nearest centre
            D=np.sort(D,axis=1)
            xt=D[:,0]
            pdf=(xt)/np.sum(xt)
            cdf=np.cumsum(pdf)
            tmp=np.random.auniform(0,1,1)
            tmp2=np.where(cdf>=tmp)[0][0]
            c[i,:]=X[tmp2,:]
        return c
    #Step 1: assign instances from dataset in must link and cannot link group
    ml=np.ones(num)
    cl=np.ones(num)
    for i in range(num):
        ml[i]=np.sum(np.sum(np.isin(must_lnk,X[i,:]),axis=1)==dim)
        cl[i]=np.sum(np.sum(np.isin(cannot_lnk,X[i,:]),axis=1)==dim)
    #initilising labels
    print(type(cl))
    Y=np.ones(num)*(np.inf)
    Y[ml==1]=0
    tmp_cnt=0
    for i in range(num):
        if cl[i]==1:
            tmp_cnt=tmp_cnt+1
            Y[i]=tmp_cnt
    #Step 2: Initialise centres
    C=np.vstack((cannot_lnk,must_link[0,:].reshape(1,-1)))
    if (tmp_cnt+1)>=K:
        K=tmp_cnt+1
    else:
        cd1=ml==0
        cd2=cl==0
        cd=np.all(np.hstack((cd1.reshape(-1,1),cd2.reshape(-1,1))),axis=1)
        tX=X[cd,:]
        tC=kpp(tX,K-tmp_cnt-1)
        C=np.vstack((C,tC))
    pdist=0
    pY=np.zeros(num)
    flag=0
    while True:
        pY[:]=Y[:]
        Xi=np.repeat(X,K,axis=0)
        Xj=np.tile(C,(num,1))
        S=np.linalg.norm(Xi-Xj,axis=1).reshape(num,K)
        tmp=np.argsort(S,axis=1)
        #checking contraints can assigning the labels
        for i in range(num):
            cnd1=(ml[i]==1)
            cnd2=(cl[i]==1)
            if cnd1==False and cnd2==False:
                Y[i]=tmp[i,0]
            else:
                cl2=np.unique(Y[cl==1])
                if cnd1==False:
                    if np.sum(np.isin(cl2,tmp[i,0]))==1:
                        Y[i]==np.nan
                    else:
                        Y[i]=tmp[i,0]
                else:
                    if np.sum(np.isin(cl2,tmp[i,0]))==0:
                        Y[i]==np.nan
                    else:
                        Y[i]=tmp[i,0]
                
        dist=np.sum(Y==pY)
        print(dist)
        if np.abs(dist-pdist)==0:
            flag+=1
        else:
            flag=0
        if flag==3:
            break
        pdist=dist
        for i in range(K):
            tx=X[i==Y,:]
            C[i,:]=np.mean(tx,axis=0)
        pdist=dist
    return(X,Y)

