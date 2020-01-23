#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy import var,sympify
#import sympy
from sympy.matrices import Matrix,zeros
import sympy


# # KNN

# In[3]:


def KNN(x_train,y_train,x_test,k,*arg):
    n1,d=np.shape(x_train)
    n2,d=np.shape(x_test)
    y_test=np.zeros(n2)
    for i in range(n2):
        dist=np.zeros(n1)
        for j in range(n1):
            if arg[0]=='cityblock':
                dist[j]=sp.spatial.distance.cityblock(x_test[i,:],x_train[j,:])
            elif arg[0]=='minkowsky':
                dist[j]=sp.spatial.distance.minkowsky(x_test[i,:],x_train[j,:])
            elif arg[0]=='hamming':
                dist[j]=sp.spatial.distance.hamming(x_test[i,:],x_train[j,:])
            else:
                dist[j]=sp.spatial.distance.euclidean(x_test[i,:],x_train[j,:])
        dist=np.argsort(dist)
        tmp=dist[0:k]
        tmp2=y_train[tmp]
        temp3=sp.stats.mode(tmp2)
        y_test[i]=temp3[0]
    return y_test


# # Logistic Regression

# In[9]:


def logistic_reg(fun,arg_x,arg_c,x_train,y_train,x_test,eta,min_itr):
    #parameters
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    n_train,_=np.shape(x_train)
    n_test,_=np.shape(x_test)
    y_train=np.reshape(y_train,(np.size(y_train),1))
    cofmat=[]
    C=np.random.random((n_ac,1))
    CP=np.zeros_like(C)
    def trns_X(xt,fun,arg_x,arg_c):
        nt,_=np.shape(xt)
        X=np.zeros((nt,n_ac))
        for i in range(nt):
            xs=xt[i,:]
            for j in range(n_ax):
                if j==0:
                    tmpx=f.subs([(arg_x[j],xs[j])])
                else:
                    tmpx=tmpx.subs([(arg_x[j],xs[j])])
            q=sympy.linear_eq_to_matrix((tmpx),cofmat)
            X[i,:]=np.array(q[0]).astype(np.float64)
        return X
    def sigmoid(x_loc):
        x_loc=np.array(x_loc).astype(np.float64)
        return 1/(1+np.exp(-x_loc))
    def evaluate_func_t(f_loc,thes):
        for j in range(n_ac):
            if j==0:
                tmpf=f_loc.subs([(arg_c[j],thes[j,0])])
            else:
                tmpf=tmpf.subs([(arg_c[j],thes[j,0])])
        return tmpf
    def p2c(pb):
        cls=np.zeros_like(pb)
        for i in range(n_train):
            if pb[i]>=0.5:
                cls[i]=1
        return cls
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    # Step 1:declaring the constant and dependent variables as var of SYMPY
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # Step 2: declaring function
    f=sympify(fun)
    # Step 3: applying kernel on x
    X_train=trns_X(x_train,fun,arg_x,arg_c)
    X_test=trns_X(x_test,fun,arg_x,arg_c)
    # Step 4: Gradient descent
    itr=0
    while True:
        print("Itr=",itr)
        itr+=1
        CP[:]=C[:]
        s=sigmoid(np.dot(X_train,C))
        tmp=(s-y_train)*X_train
        tmp2=np.sum(tmp,axis=0)
        tmp2=tmp2*eta/n_train
        tmp2=np.reshape(tmp2,(np.size(tmp2),1))
        C=CP-tmp2
        dist=np.linalg.norm(C-CP,ord='fro')
        print(dist)
        if dist==0:
            flag=flag+1
        else:
            flag=0
        if flag>=10:
            break
        if itr>=min_itr:
            break
    # Step 5: prediction
    y_test=sigmoid(np.dot(X_test,C))
    y_lbl=p2c(y_test)
    return(y_lbl,y_test,C)


# # Logistic Regression L2 Regularised

# In[2]:


def logistic_ridge_reg(fun,arg_x,arg_c,x_train,y_train,x_test,eta,lam,min_itr):
    #parameters
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    n_train,_=np.shape(x_train)
    n_test,_=np.shape(x_train)
    y_train=np.reshape(y_train,(np.size(y_train),1))
#     y_test=np.zeros((n_test,1))
    cofmat=[]
    C=np.random.random((n_ac,1))
    CP=np.zeros_like(C)
    def trns_X(xt,fun,arg_x,arg_c):
        nt,_=np.shape(xt)
        X=np.zeros((nt,n_ac))
        for i in range(nt):
            xs=xt[i,:]
            for j in range(n_ax):
                if j==0:
                    tmpx=f.subs([(arg_x[j],xs[j])])
                else:
                    tmpx=tmpx.subs([(arg_x[j],xs[j])])
            q=sympy.linear_eq_to_matrix((tmpx),cofmat)
            X[i,:]=np.array(q[0]).astype(np.float64)
        return X
    def sigmoid(x_loc):
        x_loc=np.array(x_loc).astype(np.float64)
        return 1/(1+np.exp(-x_loc))
    def evaluate_func_t(f_loc,thes):
        for j in range(n_ac):
            if j==0:
                tmpf=f_loc.subs([(arg_c[j],thes[j,0])])
            else:
                tmpf=tmpf.subs([(arg_c[j],thes[j,0])])
        return tmpf
    def p2c(pb):
        cls=np.zeros_like(pb)
        for i in range(n_train):
            if pb[i]>=0.5:
                cls[i]=1
        return cls
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    # Step 1:declaring the constant and dependent variables as var of SYMPY
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # Step 2: declaring function
    f=sympify(fun)
    # Step 3: applying kernel on x
    X_train=trns_X(x_train,fun,arg_x,arg_c)
    X_test=trns_X(x_test,fun,arg_x,arg_c)
    # Step 4: Gradient descent
    itr=0
    while True:
        print("Itr=",itr)
        itr+=1
        CP[:]=C[:]
        s=sigmoid(np.dot(X_train,C))
        tmp=(s-y_train)*X_train
        tmp2=np.sum(tmp,axis=0)
        tmp2=tmp2*eta/n_train
        tmp2=np.reshape(tmp2,(np.size(tmp2),1))
        C=(1-eta*lam/n_train)*CP-tmp2
        dist=np.linalg.norm(C-CP,ord='fro')
        print(dist)
        if dist==0:
            flag=flag+1
        else:
            flag=0
        if flag>=10:
            break
        if itr>=min_itr:
            break
    # Step 5: prediction
    y_test=sigmoid(np.dot(X_test,C))
    y_lbl=p2c(y_test)
    return(y_lbl,y_test,C)


# #  Logistic Regression L1 Regularised

# In[1]:


def logistic_lasso_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam,min_itr):
    #parameters
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    n_train,_=np.shape(x_train)
    n_test,_=np.shape(x_train)
    y_train=np.reshape(y_train,(np.size(y_train),1))
#     y_test=np.zeros((n_test,1))
    cofmat=[]
    C=np.random.random((n_ac,1))
    CP=np.zeros_like(C)
    def trns_X(xt,fun,arg_x,arg_c):
        nt,_=np.shape(xt)
        X=np.zeros((nt,n_ac))
        for i in range(nt):
            xs=xt[i,:]
            for j in range(n_ax):
                if j==0:
                    tmpx=f.subs([(arg_x[j],xs[j])])
                else:
                    tmpx=tmpx.subs([(arg_x[j],xs[j])])
            q=sympy.linear_eq_to_matrix((tmpx),cofmat)
            X[i,:]=np.array(q[0]).astype(np.float64)
        return X
    def sigmoid(x_loc):
        x_loc=np.array(x_loc).astype(np.float64)
        return 1/(1+np.exp(-x_loc))
    def evaluate_func_t(f_loc,thes):
        for j in range(n_ac):
            if j==0:
                tmpf=f_loc.subs([(arg_c[j],thes[j,0])])
            else:
                tmpf=tmpf.subs([(arg_c[j],thes[j,0])])
        return tmpf
    def p2c(pb):
        cls=np.zeros_like(pb)
        for i in range(n_train):
            if pb[i]>=0.5:
                cls[i]=1
        return cls
    def soft_thr(pj,lm,zj):
        if pj<(-lm):
            return (pj+lm)/zj
        elif pj>lm:
            return (pj-lm)/zj
        else:
            return 0
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    # Step 1:declaring the constant and dependent variables as var of SYMPY
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # Step 2: declaring function
    f=sympify(fun)
    # Step 3: applying kernel on x
    X_train=trns_X(x_train,fun,arg_x,arg_c)
    X_test=trns_X(x_test,fun,arg_x,arg_c)
    # Step 4: Coordinate descent
    itr=0
    while True:
        print("Itr=",itr)
        itr+=1
        CP[:]=C[:]
        s=sigmoid(np.dot(X_train,C))
        for j in range(n_ac):
            Xj=X_train[:,j].reshape(-1,1)
            Zj=np.linalg.norm(Xj)**2
            tPj=Xj*(y_train-s+C[j,:]*Xj)
            Pj=np.sum(tPj)
            C[j,:]=soft_thr(Pj,lam,Zj)
        dist=np.linalg.norm(C-CP,ord='fro')
        print(dist)
        if dist==0:
            flag=flag+1
        else:
            flag=0
        if flag>=10:
            break
        if itr>=min_itr:
            break
    # Step 5: prediction
    y_test=sigmoid(np.dot(X_test,C))
    y_lbl=p2c(y_test)
    return(y_lbl,y_test,C)


# #  Logistic Regression L1,L2 Regularised

# In[ ]:


def logistic_elasticnet_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam1,lam2,min_itr):
    #parameters
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    n_train,_=np.shape(x_train)
    n_test,_=np.shape(x_train)
    y_train=np.reshape(y_train,(np.size(y_train),1))
#     y_test=np.zeros((n_test,1))
    cofmat=[]
    C=np.random.random((n_ac,1))
    CP=np.zeros_like(C)
    def trns_X(xt,fun,arg_x,arg_c):
        nt,_=np.shape(xt)
        X=np.zeros((nt,n_ac))
        for i in range(nt):
            xs=xt[i,:]
            for j in range(n_ax):
                if j==0:
                    tmpx=f.subs([(arg_x[j],xs[j])])
                else:
                    tmpx=tmpx.subs([(arg_x[j],xs[j])])
            q=sympy.linear_eq_to_matrix((tmpx),cofmat)
            X[i,:]=np.array(q[0]).astype(np.float64)
        return X
    def sigmoid(x_loc):
        x_loc=np.array(x_loc).astype(np.float64)
        return 1/(1+np.exp(-x_loc))
    def evaluate_func_t(f_loc,thes):
        for j in range(n_ac):
            if j==0:
                tmpf=f_loc.subs([(arg_c[j],thes[j,0])])
            else:
                tmpf=tmpf.subs([(arg_c[j],thes[j,0])])
        return tmpf
    def p2c(pb):
        cls=np.zeros_like(pb)
        for i in range(n_train):
            if pb[i]>=0.5:
                cls[i]=1
        return cls
    def soft_thr(pj,lm,zj):
        if pj<(-lm):
            return (pj+lm)/zj
        elif pj>lm:
            return (pj-lm)/zj
        else:
            return 0
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    # Step 1:declaring the constant and dependent variables as var of SYMPY
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # Step 2: declaring function
    f=sympify(fun)
    # Step 3: applying kernel on x
    X_train=trns_X(x_train,fun,arg_x,arg_c)
    X_test=trns_X(x_test,fun,arg_x,arg_c)
    # Step 4: Coordinate descent
    itr=0
    while True:
        print("Itr=",itr)
        itr+=1
        CP[:]=C[:]
        s=sigmoid(np.dot(X_train,C))
        for j in range(n_ac):
            Xj=X_train[:,j].reshape(-1,1)
            Zj=np.linalg.norm(Xj)**2+lam2
            tPj=Xj*(y_train-s+C[j,:]*Xj)
            Pj=np.sum(tPj)
            C[j,:]=soft_thr(Pj,lam1,Zj)
        dist=np.linalg.norm(C-CP,ord='fro')
        print(dist)
        if dist==0:
            flag=flag+1
        else:
            flag=0
        if flag>=10:
            break
        if itr>=min_itr:
            break
    # Step 5: prediction
    y_test=sigmoid(np.dot(X_test,C))
    y_lbl=p2c(y_test)
    return(y_lbl,y_test,C)


# # LDA

# In[4]:


def LDA(x_train,x_test,y_train):
    pts,dim=np.shape(x_train)
    cls=np.unique(y_train)
    k=np.size(cls)
    df=np.zeros((k,pts))
    def prior(x_train,y_train):
        pik=[]
        for i in cls:
            ind=(y_train==i)
            xc=x_train[ind,:]
            tmp,_=np.shape(xc)
            pik.append(tmp)
        sm=np.sum(pik)
        pik=pik/sm
        return(pik)
    def mean_x(x):
        m=np.zeros((k,dim))
        for i in range(k):
            ind=(y_train==cls[i])
            xc=x_train[ind,:]
            m[i,:]=np.mean(xc,axis=0)
        return(m)
    def covar_x(x):
        mutmp=np.mean(x,axis=0)
        x_t=x-mutmp
        c=(1/(pts-1))*(np.dot(x_t.T,x_t))
        return c
    #step 1: prior distribution
    pi=prior(x_train,y_train)
    #step 2: mean for each class
    mu=mean_x(x_train)
    #step 3: covariance of the data
    S=covar_x(x_train)
    Si=np.linalg.inv(S)
    #step 4: posterior distribution(not required, for demonstration purpose only)
    for i in range(k):
        for j in range(pts):
            mu_v=np.reshape(mu[i,:],(dim,1))
            x_v=np.reshape(x_train[j,:],(dim,1))
            t1=np.log(pi[i])
            t2=-0.5*(np.dot(np.dot(mu_v.T,Si),mu_v))
            t3=np.dot(np.dot(x_v.T,Si),mu_v)
            df[i,j]=t1+t2+t3
    #step 5: prediction
    n_tst,_=np.shape(x_test)
    y=np.zeros((k,n_tst))
    for i in range(k):
        for j in range(n_tst):
            mu_v=np.reshape(mu[i,:],(dim,1))
            x_v=np.reshape(x_test[j,:],(dim,1))
            t1=np.log(pi[i])
            t2=-0.5*(np.dot(np.dot(mu_v.T,Si),mu_v))
            t3=np.dot(np.dot(x_v.T,Si),mu_v)
            y[i,j]=t1+t2+t3
    y_lbl=np.argmax(y,axis=0)
    return(y,y_lbl)


# # QDA

# In[ ]:


def QDA(x_train,x_test,y_train):
    pts,dim=np.shape(x_train)
    cls=np.unique(y_train)
    k=np.size(cls)
    df=np.zeros((k,pts))
    def prior(x_train,y_train):
        pik=[]
        for i in cls:
            ind=(y_train==i)
            xc=x_train[ind,:]
            tmp,_=np.shape(xc)
            pik.append(tmp)
        sm=np.sum(pik)
        pik=pik/sm
        return(pik)
    def mean_x(x):
        m=np.zeros((k,dim))
        for i in range(k):
            ind=(y_train==cls[i])
            xc=x_train[ind,:]
            m[i,:]=np.mean(xc,axis=0)
        return(m)
    def covar_x(xt,yt):
        sk=np.zeros((k,dim,dim))
        for i in range(k):
            lbl=(yt==cls[i])
            x=xt[lbl,:]
            mutmp=np.mean(x,axis=0)
            x_t=x-mutmp
            sk[i,:,:]=(1/(pts-1))*(np.dot(x_t.T,x_t))
        return sk
    #step 1: prior distribution
    pi=prior(x_train,y_train)
    #step 2: mean for each class
    mu=mean_x(x_train)
    #step 3: covariance of the data
    Sk=covar_x(x_train,y_train)
    Ski=np.zeros((k,dim,dim))
    for i in range(k):
        Ski[i,:,:]=np.linalg.inv(Sk[i,:,:])
    #step 4: posterior distribution(not required, for demonstration purpose only)
    for i in range(k):
        for j in range(pts):
            mu_v=np.reshape(mu[i,:],(dim,1))
            x_v=np.reshape(x_train[j,:],(dim,1))
            t1=np.log(pi[i])
            t2=-0.5*(np.dot(np.dot(mu_v.T,Ski[i,:,:]),mu_v))
            t3=np.dot(np.dot(x_v.T,Ski[i,:,:]),mu_v)
            t4=-0.5*np.dot(np.dot(x_v.T,Ski[i,:,:]),x_v)
            t5=-0.5*(np.log(np.abs(np.linalg.det(Ski[i,:,:]))))
            df[i,j]=t1+t2+t3+t4+t5
    #step 5: prediction
    n_tst,_=np.shape(x_test)
    y=np.zeros((k,n_tst))
    for i in range(k):
        for j in range(n_tst):
            mu_v=np.reshape(mu[i,:],(dim,1))
            x_v=np.reshape(x_test[j,:],(dim,1))
            t1=np.log(pi[i])
            t2=-0.5*(np.dot(np.dot(mu_v.T,Ski[i,:,:]),mu_v))
            t3=np.dot(np.dot(x_v.T,Ski[i,:,:]),mu_v)
            t4=-0.5*np.dot(np.dot(x_v.T,Ski[i,:,:]),x_v)
            t5=-0.5*(np.log(np.abs(np.linalg.det(Ski[i,:,:]))))
            y[i,j]=t1+t2+t3+t4+t5
    y_lbl=np.argmax(y,axis=0)
    return(y,y_lbl)


# # Naive Bayes

# In[1]:


def gaussian_NB(x_train,y_train,x_test):
    def normal(x,p1,p2):
        den=np.sqrt(2*(np.pi)*(p2**2))
        num=np.exp(-((x-p1)**2)/(2*(p2**2)))
        p=num/den
        return p
    num,dim=np.shape(x_train)
    tst_pts,_=np.shape(x_test)
    cls=np.unique(y_train)
    n_cls=np.size(cls)
    #initialisation
    mu=np.zeros((n_cls,dim))
    sig=np.zeros((n_cls,dim))
    pi=np.zeros(n_cls)
    #prior Distribution
    for i in range(n_cls):
        ct=np.sum(y==cls[i])
        pi[i]=ct/num
    #likelihood parameter
    for i in range(n_cls):
        ind=(y_train==cls[i])
        cls_pts=x_train[ind,:]
        mu[i,:]=np.mean(cls_pts,axis=0)
        sig[i,:]=np.std(cls_pts,axis=0)
    #posterior distribution/testing
    y_test=np.zeros((tst_pts,n_cls))
    for k in range(n_cls):
        for i in range(tst_pts):
            tmp=1
            for j in range(dim):
                tmp=tmp*normal(x_test[i,j],mu[k,j],sig[k,j])
            y_test[i,k]=tmp*pi[k]
    #Labels
    y_tmp=np.argsort(y_test,axis=1)
    y_tmp2=y_tmp[:,-1]
    y_lbl=[cls[i] for i in y_tmp2]
    return(y_lbl,y_tmp2)


# # Support Vector Machine

# In[ ]:


def SVM(X,Y,x_test,C,*args,**kwargs):
    num,dim=np.shape(X)
    tnum,_=np.shape(x_test)
    #optimisation algorithm
    max_pass=kwargs['max_passes']
    def smo(K,Y):
        Alpha=np.zeros(num)
        B=0
        tol=0.01
        passes=0
        while(max_pass>passes):
            num_changed_alpha=0
            for i in range(num):
                fx=np.sum(Alpha.reshape(-1,1)*Y.reshape(-1,1)*K,axis=0)+B
                fxi=fx[i]
                Ei=fxi-Y[i]
                cnd1=Y[i]*Ei<-tol and Alpha[i]<=C
                cnd2=Y[i]*Ei>tol and Alpha[i]>=0
                if (cnd1==True) or (cnd2==True):
                    while True:
                        j=int(np.random.randint(0,num))
                        if j!=i:
                            break
                    fxj=fx[j]
                    Ej=fxj-Y[j]
                    old_ai=Alpha[i]
                    old_aj=Alpha[j]
                    if Y[i]!=Y[j]:
                        L=max(0,Alpha[j]-Alpha[i])
                        H=min(C,C+Alpha[j]-Alpha[i])
                    else:
                        L=max(0,Alpha[j]+Alpha[i]-C)
                        H=min(C,Alpha[j]+Alpha[i])
                    if L==H:
                        continue
                    eta=2*K[i,j]-K[i,i]-K[j,j]
                    if eta>=0:
                        continue
                    Alpha[j]=Alpha[j]-Y[j]*(Ei-Ej)/eta
                    if Alpha[j]>H:
                        Alpha[j]=H
                    elif Alpha[j]<L:
                        Alpha[j]=L
                    else:
                        pass
                    if np.abs(Alpha[j]-old_aj)<tol:
                        continue
                    Alpha[i]=Alpha[i]+Y[i]*Y[j]*(old_aj-Alpha[j])
                    cnd1=0<Alpha[i] and Alpha[i]<C
                    cnd2=0<Alpha[j] and Alpha[j]<C
                    if cnd1==True and cnd2==False:
                        b1=B-Ei-Y[i]*(Alpha[i]-old_ai)*K[i,i]-Y[j]*(Alpha[j]-old_aj)*K[i,j]
                        B=b1
                    elif cnd1==False and cnd2==True:
                        b2=B-Ej-Y[i]*(Alpha[i]-old_ai)*K[i,j]-Y[j]*(Alpha[j]-old_aj)*K[j,j]
                        B=b2
                    else:
                        b1=B-Ei-Y[i]*(Alpha[i]-old_ai)*K[i,i]-Y[j]*(Alpha[j]-old_aj)*K[i,j]
                        b2=B-Ej-Y[i]*(Alpha[i]-old_ai)*K[i,j]-Y[j]*(Alpha[j]-old_aj)*K[j,j]
                        B=(b1+b2)/2
                    num_changed_alpha=num_changed_alpha+1
            if num_changed_alpha==0:
                passes=passes+1
            else:
                passes=0
        return(Alpha,B)
    #kernels
    def linear(x,y):
        return(np.dot(x,y.T))
    def poly(x,y):
        tmp=(np.dot(x,y.T)*kwargs['gamma']+kwargs['c0'])**(kwargs['degree'])
        return(tmp)
    def rbf(x,y):
        tmp=np.linalg.norm(x-y,axis=1)**2
        return(np.exp(-tmp*(kwargs['gamma'])))
    #Step 1: Transforming input with kernel
    if args[0]=='linear':
        K=linear(X,X).reshape(num,num)
        tK=linear(X,x_test).reshape(num,tnum)
    elif args[0]=='poly':
        K=poly(X,X).reshape(num,num)
        tK=poly(X,x_test).reshape(num,tnum)
    else:
        Xi=np.repeat(X,num,axis=0)
        Xj=np.tile(X,(num,1))
        tXj=np.tile(x_test,(tnum,1))
        K=rbf(Xi,Xj).reshape(num,num)
        tK=rbf(Xi,tXj).reshape(num,tnum)
    #Step 2: Sequential minimal optimization
    Alpha,B=smo(K,Y)
    #Step 3: Predictions
    ty_test=np.sum(Alpha.reshape(-1,1)*Y.reshape(-1,1)*tK,axis=0)+B
    y_test[ty_test>=0]=1
    y_test[ty_test<0]=-1
    return(y_test)


# # Perceptron

# In[ ]:


def perceptron(X_train,y_train,X_test,**kwargs):
    n_train,dim=np.shape(X_train)
    W=np.random.rand(dim,1)
    b=np.random.rand()
    old_W=np.random.rand(dim,1)
    old_b=np.random.rand()
    flag=0
    try:
        trshld=kwargs['eps']
    except:
        trshld=0.0001
    try:
        flg_count=kwargs['stable']
    except:
        flg_count=10
    try:
        min_err=10
        best_W=np.random.rand(dim,1)
        best_b=np.random.rand()
        t_itr=kwargs['itr']
    except:
        pass
    
    itr=-1
    #Training
    while True:
        itr+=1
        old_W[:]=W[:]
        old_b=b
        for i in range(n_train):
            Xi=X_train[i,:].reshape(-1,1)
            yp=np.dot(W.T,Xi)+b
            if yp>0:
                Y=1
            else:
                Y=0
            error=y_train[i]-Y
            W=W+error*Xi
            b=b+error
        epsilon=np.abs(np.linalg.norm(old_W-W,ord='fro')+old_b-b)
        if min_err>np.sum(np.abs(error)):
            best_W=W[:]
            best_b=b
            min_err=error
        if epsilon<trshld:
            flag=flag+1
        else:
            flag=0
        if flag==flg_count:
            break
        try:
            if t_itr<=itr:
                W=best_W
                b=best_b
                break
        except:
            pass
    #Prediction
    n_test,_=np.shape(X_test)
    yp=np.dot(W.T,X_test.T)+b
    Y_hat=(yp>0)
    return(Y_hat.flatten(),W,b)


# # RBF Neural Network

# In[ ]:


def RBFN(X_train,y_train,X_test,no_h):
    num1,D=np.shape(X_train)
    num2,_=np.shape(X_test)
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
            tmp=np.random.uniform(0,1,1)
            tmp2=np.where(cdf>=tmp)[0][0]
            c[i,:]=X[tmp2,:]
        return c
    #Step 1: Centre of gaussians
    C=kpp(X_train,no_h)
    #Step 2: Hidden layer activation
    Xi=np.repeat(X_train,no_h,axis=0)
    Xj=np.tile(C,(num1,1))
    S=np.exp(-(np.linalg.norm(Xi-Xj,axis=1)**2)/2).reshape(num1,no_h)
    #Step 3: Closed form solution, one vs all
    W=[]
    for i in range(np.size(np.unique(y_train))):
        pS=np.linalg.pinv(S)
        tmp_y=(y_train==i)
        W.append((np.dot(pS,tmp_y)).T)
    #Prediction
    Xi=np.repeat(X_test,no_h,axis=0)
    Xj=np.tile(C,(num2,1))
    S=np.exp(-(np.linalg.norm(Xi-Xj,axis=1)**2)/2).reshape(num2,no_h)
    tmp_y=np.zeros((num2,np.size(np.unique(y_train))))
    for i in range(np.size(np.unique(y_train))):
        tmp_y[:,i]=np.dot(S,W[i].T)
    y_test=np.argmax(tmp_y,axis=1)
    return(y_test)

