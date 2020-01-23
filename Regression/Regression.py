#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
from sympy import var,sympify
import sympy
from sympy.matrices import Matrix,zeros
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


# # Linear Regression

# In[3]:


def linear_reg(fun,arg_x,arg_c,x_train,y_train,x_test):
    # declaring the constant and dependent variables as var of SYMPY
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    cofmat=[]
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # declaring function
    f=sympify(fun)
    #fetching training data and feeding function
    n_train,n_ax=np.shape(x_train)
    mat_fun=[]
    for i in range(n_train):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_train[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_train[i,j])])
        mat_fun.append(tmp)
    #The "A" Matrix
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    A=np.array(At).astype(np.float64)
    # Pesudo inverse of A
    pA=np.linalg.pinv(A)
    #coefficent prediction for the model
    C=np.dot(pA,y_train)
    #predictions
    n_test,n_ax=np.shape(x_test)
    mat_fun=[]
    for i in range(n_test):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_test[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_test[i,j])])
        mat_fun.append(tmp)
    #
    cofmat=[]
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    y_test=np.dot(At,C.T)
    return(y_test,C)


# # Linear regression  L2 Regularised

# In[1]:


def linear_ridge_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam):
    # declaring the constant and dependent variables as var of SYMPY
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    cofmat=[]
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # declaring function
    f=sympify(fun)
    #fetching training data and feeding function
    n_train,n_ax=np.shape(x_train)
    mat_fun=[]
    for i in range(n_train):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_train[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_train[i,j])])
        mat_fun.append(tmp)
    #The "A" Matrix
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    A=np.array(At).astype(np.float64)
    # Pesudo inverse of A
    p1=np.dot(A.T,A)
    nv,_=np.shape(p1)
    p2=np.identity(nv)
    p2[0,0]=0
    p3=np.linalg.inv(p1+p2*(lam**2))
    pA=np.dot(p3,A.T)   
    #coefficent prediction for the model
    C=np.dot(pA,y_train)
    #predictions
    n_test,n_ax=np.shape(x_test)
    mat_fun=[]
    for i in range(n_test):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_test[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_test[i,j])])
        mat_fun.append(tmp)
    #
    cofmat=[]
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    y_test=np.dot(At,C.T)
    return(y_test,C)


# # Linear regression  L1 Regularised

# In[ ]:


def linear_lasso_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam,min_itr):
    #parameters
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    n_train,_=np.shape(x_train)
    n_test,_=np.shape(x_test)
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
    def evaluate_func_t(f_loc,thes):
        for j in range(n_ac):
            if j==0:
                tmpf=f_loc.subs([(arg_c[j],thes[j,0])])
            else:
                tmpf=tmpf.subs([(arg_c[j],thes[j,0])])
        return tmpf
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
        s=np.dot(X_train,C)
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
    y_test=np.dot(X_test,C)
    return(y_test,C)


# #  Linear Regression L1,L2 Regularised

# In[ ]:


def linear_elasticnet_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam1,lam2,min_itr):
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
        s=np.dot(X_train,C)
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
    y_test=np.dot(X_test,C)
    return(y_test,C)


# # MLE Linear Regression

# In[ ]:


def mle_linear_reg(fun,arg_x,arg_c,x_train,y_train,x_test):
    # declaring the constant and dependent variables as var of SYMPY
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    cofmat=[]
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # declaring function
    f=sympify(fun)
    #fetching training data and feeding function
    n_train,n_ax=np.shape(x_train)
    mat_fun=[]
    for i in range(n_train):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_train[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_train[i,j])])
        mat_fun.append(tmp)
    #The "A" Matrix
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    A=np.array(At).astype(np.float64)
    # Pesudo inverse of A
    pA=np.linalg.pinv(A)
    #coefficent prediction for the model
    C=np.dot(pA,y_train)
    #predictions of test values
    n_test,n_ax=np.shape(x_test)
    mat_fun=[]
    for i in range(n_test):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_test[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_test[i,j])])
        mat_fun.append(tmp)
    #
    cofmat=[]
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    qq=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=qq[0]
    y_test=np.dot(At,C)
    #prediction of variance
    yhat=np.dot(At,C)
    tp1=y_train-yhat
    tp2=np.dot(tp1.T,tp1)/n_train
    sigma=np.sqrt(np.array(tp2,dtype='float64'))
    return(y_test,sigma,C)


# # Bayesian Ridge Regression

# In[ ]:


def bayesian_ridge_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam):
    # declaring the constant and dependent variables as var of SYMPY
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    cofmat=[]
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # declaring function
    f=sympify(fun)
    #fetching training data and feeding function
    n_train,n_ax=np.shape(x_train)
    mat_fun=[]
    for i in range(n_train):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_train[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_train[i,j])])
        mat_fun.append(tmp)
    #The "A" Matrix
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    A=np.array(At).astype(np.float64)
    # Pesudo inverse of A
    p1=np.dot(A.T,A)
    nv,_=np.shape(p1)
    p2=np.identity(nv)
    p2[0,0]=0
    p3=np.linalg.inv(p1+p2*lam)
    pA=np.dot(p3,A.T)   
    #coefficent prediction for the model
    C=np.dot(pA,y_train)
    #predictions
    n_test,n_ax=np.shape(x_test)
    mat_fun=[]
    for i in range(n_test):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_test[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_test[i,j])])
        mat_fun.append(tmp)
    #
    cofmat=[]
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    qq=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=qq[0]
    y_test=np.dot(At,C)
    #prediction of variance
    tp1=y_train-np.dot(q[0],C)
    tp2=np.dot(tp1.T,tp1)/n_train
    sigma2=np.array(tp2,dtype='float32')
    tp1=y_train-np.dot(q[0],C)
    tp2=np.dot(tp1.T,tp1)/n_train
    sigma2=np.sqrt(np.array(tp2,dtype='float32'))
    sigma=np.ones((n_test,1))*sigma2
    #########
    V=np.linalg.inv(p2+p1/sigma2)
    for i in range(n_test):
        xt=np.reshape(At[i,:],(1,-1))
        tp1=np.dot(xt,V)
        sigma[i,:]=sigma[i,:]+np.dot(tp1,xt.T)
    sigma=np.sqrt(sigma)
    return(y_test,sigma,C)


# # Gaussian Processes

# In[ ]:


def GP_regressor(x_train,y_train,x_test,sig_y,sig_f,l):
    num,dim=np.shape(x_train)
    num2,dim=np.shape(x_test)
    def kernel(x,xp):
        return (sig_f**2)*np.exp(-(x-xp)**2/2*l**2)
    #gaussian marginals
    #mu is zero
    #(1,1)
    Xi=np.repeat(x_train,num,axis=0)
    Xj=np.tile(x_train,(num,1))
    Kt=kernel(Xi,Xj).reshape(num,num)
    Ky=Kt+(sig_y**2)*np.identity(num)
    #(1,2)(2,1)
    Xi=np.repeat(x_train,num2,axis=0)
    Xj=np.tile(x_test,(num,1))
    Kp=kernel(Xi,Xj).reshape(num,num2)
    #(2,2)
    Xi=np.repeat(x_test,num2,axis=0)
    Xj=np.tile(x_test,(num2,1))
    #gaussian joint distribution
    Kpp=kernel(Xi,Xj).reshape(num2,num2)
    p1=np.hstack((Ky,Kp))
    p2=np.hstack((Kp.T,Kpp))
    K=np.vstack((p1,p2))
    #step 2: gaussian conditionals
    mu=np.dot(np.dot(Kp.T,np.linalg.inv(Ky)),y_train)
    S=Kpp-np.dot(np.dot(Kp.T,np.linalg.inv(Ky)),Kp)
    return(mu,S)


# # RANSAC

# In[2]:


def ransac(x_train,y_train,x_test,min_pts,sigma,min_iter):
    num,dim=np.shape(x_train)
    itr=0
    inlier=np.empty((num,min_iter))
    while(itr<min_iter):
        #Step 1: Randomly select min points for regression
        ind=random.sample(range(0,num),min_pts)
        xin=x_train[ind,:]
        yin=y_train[ind,:]
        #step 2: Perform regression
        tmp=np.ones((min_pts,1))
        Xin=np.hstack((tmp,xin))
        iX=np.linalg.pinv(Xin)
        Cin=np.dot(iX,yin)
        #Step 3: inliers
        X_train=np.hstack((np.ones((num,1)),x_train))
        Din=np.abs((np.dot(X_train,Cin)-y_train)/np.sqrt(1+np.linalg.norm(Cin)**2))
        cnd=(Din<=sigma)
        inlier[:,itr]=cnd.flatten()
        itr=itr+1
    #Step 4: model selection
    itrid=np.argmax(np.sum(inlier,axis=0))
    ind=inlier[:,itrid]
    ind=(ind>=1)
    num=np.sum(ind)
    xin=x_train[ind,:]
    yin=y_train[ind,:]
    tmp=np.ones((num,1))
    Xin=np.hstack((tmp,xin))
    iX=np.linalg.pinv(Xin)
    Cin=np.dot(iX,yin)
    #Step 5: Prediction
    num,_=np.shape(x_test)
    tmp=np.ones((num,1))
    Xt=np.hstack((tmp,x_test))
    y_test=np.dot(Xt,Cin)
    return(xin,yin,Cin,y_test)


# # Nadaraya Watson Regression

# In[ ]:


def nadaraya_watson(X_train,Y_train,X_test,**kwargs):
    n_pts1,dim=np.shape(X_train)
    n_pts2,_=np.shape(X_test)
    Y_test=np.zeros(n_pts2)
    def bw(strg,x):
        try:
            n,d=np.shape(x)
        except:
            d=1
            n=np.size(x)
        sgm=np.std(x,axis=0)
        H=np.zeros((d,d))
        if strg=='silverman':
            for i in range(d):
                H[i,i]=(((4/(d+2))**(1/d+4))*(n**(-1/(d+4)))*(sgm[i]))**2
            return H
        elif strg=='scott':
            for i in range(d):
                H[i,i]=((n**(-1/(d+4)))*(sgm[i]))**(2)
            return H
        else:
            print('Unrecognised method for bandweidth selection')
    def kernel(H,x):
        try:
            n,d=np.shape(x)
        except:
            d=1
            n=np.size(x)
        y1=((2*np.pi)**(-d/2))
        y2=((np.linalg.det(H))**(-1/2))
        y31=np.dot(np.transpose(x),np.linalg.inv(H))
        y3=np.exp((-1/2)*np.dot(y31,x))
        y=y1*y2*y3
        return y
    #Step 1: banwidth calculation
    try:
        strg=kwargs['method']
        H=bw(strg,X_train)
    except:
        H=kwargs['BW']
    print('BW',H)
    #Step 2: predictions
    for i in range(n_pts2):
        num=0
        den=0
        for j in range(n_pts1):
            num=num+kernel(H,X_test[i,:]-X_train[j,:])*Y_train[j]
            den=den+kernel(H,X_test[i,:]-X_train[j,:])
        Y_test[i]=num/den
    return(Y_test)


# # Local Regression

# In[ ]:


def LLR(X_train,Y_train,X_test,tau):
    num1,dim=np.shape(X_train)
    num2,_=np.shape(X_test)
    Y_test=np.zeros(num2)
    def gauss(xi,xj):
        return np.exp(-(xi-xj)**2/2*tau**2)
    for i in range(num2):
        #Step 1: X
        dX=X_train
        col1=np.ones((num1,1))
        X=np.hstack((col1,dX))
        #Step 2: W
        w=gauss(X_train,X_test[i,:]).flatten()
        W=np.diag(w)
        #Step 3: closed form solution
        p1=np.dot(X.T,W)
        p2=np.linalg.inv(np.dot(p1,X))
        p3=np.dot(p2,p1)
        beta=np.dot(p3,Y_train.reshape(-1,1))
        #Step4: Prediction
        xt=X_test[i,:].reshape(1,-1)
        xt=np.hstack((np.array([[1]]),xt))
        Y_test[i]=np.dot(xt,beta)
    return(Y_test)


# # KNN Regression

# In[1]:


def KNN_reg(x_train,y_train,x_test,K,P):
    num1,dim=np.shape(x_train)
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


# # Perceptron/ADALINE Regression

# In[1]:


def :
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
            Y=yp
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
            if t_itr==itr:
                W=best_W
                b=best_b
                break
        except:
            pass
    #Prediction
    yp=np.dot(W.T,X_test.T)+b
    Y_hat=yp
    return(Y_hat.flatten(),W,b)


# # Chebyshev-FLNN

# In[5]:


def CFLNN(X_train,y_train,X_test,l,eta,**kwargs):
    def chebyshev(l,x):
        pts=np.size(x)
        Yc=np.ones((l,pts))
        try:
            if kwargs['kind']==1:
                Yc[1,:]=x
            if kwargs['kind']==2:
                Yc[1,:]=2*x
        except:
            Yc[1,:]=x
        for i in range(2,l):
            Yc[i,:]=2*x*Yc[i-1,:]-Yc[i-2,:]
        return(Yc)
    n_train,dim=np.shape(X_train)
    n_test,_=np.shape(X_test)
    # Function Expansion Block
    T_train=np.ones((n_train,dim*(l-1)+1))
    T_test=np.ones((n_test,dim*(l-1)+1))
    for i in range(dim):
        tmp_train_Yc=chebyshev(l,X_train[:,i]).T
        tmp_test_Yc=chebyshev(l,X_test[:,i]).T
        T_train[:,(i*(l-1)+1):(i*(l-1)+l)]=tmp_train_Yc[:,1:]
        T_test[:,(i*(l-1)+1):(i*(l-1)+l)]=tmp_test_Yc[:,1:]
    W=np.random.rand(dim*(l-1)+1,1)
    old_W=np.random.rand(dim*(l-1)+1,1)
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
        best_W=np.random.rand(dim*(l-1)+1,1)
        t_itr=kwargs['itr']
    except:
        pass
    
    itr=-1
    #Training
    while True:
        itr+=1
        old_W[:]=W[:]
        for i in range(n_train):
            Xi=T_train[i,:].reshape(-1,1)
            MA=np.mean(Xi)
            yp=np.dot(W.T,Xi)+MA
            Y=yp
            error=y_train[i]-Y
            W=W+eta*error*Xi
        epsilon=np.abs(np.linalg.norm(old_W-W,ord='fro'))
        if min_err>np.sum(np.abs(error)):
            best_W=W[:]
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
                break
        except:
            pass
    #Prediction
    MA=np.mean(T_test,axis=1)
    yp=np.dot(W.T,T_test.T)+MA
    Y_hat=yp
    return(Y_hat.flatten(),W)


# # Legendre-FLNN

# In[4]:


def LeFLNN(X_train,y_train,X_test,l,eta,**kwargs):
    def legendre(l,x):
        pts=np.size(x)
        Yc=np.ones((l,pts))
        Yc[1,:]=x
        for i in range(2,l):
            Yc[i,:]=((2*(i-1)+1)*x*Yc[i-1,:]-(i-1)*Yc[i-2,:])/i
        return(Yc)
    n_train,dim=np.shape(X_train)
    n_test,_=np.shape(X_test)
    # Function Expansion Block
    T_train=np.ones((n_train,dim*(l-1)+1))
    T_test=np.ones((n_test,dim*(l-1)+1))
    for i in range(dim):
        tmp_train_Yc=legendre(l,X_train[:,i]).T
        tmp_test_Yc=legendre(l,X_test[:,i]).T
        T_train[:,(i*(l-1)+1):(i*(l-1)+l)]=tmp_train_Yc[:,1:]
        T_test[:,(i*(l-1)+1):(i*(l-1)+l)]=tmp_test_Yc[:,1:]
    W=np.random.rand(dim*(l-1)+1,1)
    old_W=np.random.rand(dim*(l-1)+1,1)
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
        best_W=np.random.rand(dim*(l-1)+1,1)
        t_itr=kwargs['itr']
    except:
        pass
    
    itr=-1
    #Training
    while True:
        itr+=1
        old_W[:]=W[:]
        for i in range(n_train):
            Xi=T_train[i,:].reshape(-1,1)
            MA=np.mean(Xi)
            yp=np.dot(W.T,Xi)+MA
            Y=yp
            error=y_train[i]-Y
            W=W+eta*error*Xi
        epsilon=np.abs(np.linalg.norm(old_W-W,ord='fro'))
        if min_err>np.sum(np.abs(error)):
            best_W=W[:]
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
                break
        except:
            pass
    #Prediction
    MA=np.mean(T_test,axis=1)
    yp=np.dot(W.T,T_test.T)+MA
    Y_hat=yp
    return(Y_hat.flatten(),W)


# # Laguerre-FLNN 

# In[3]:


def LFLNN(X_train,y_train,X_test,l,eta,**kwargs):
    def laguerre(l,x):
        pts=np.size(x)
        Yc=np.ones((l,pts))
        Yc[1,:]=1-x
        for i in range(2,l):
            Yc[i,:]=((2*(i-1)+1-x)*Yc[i-1,:]-(i-1)*Yc[i-2,:])/i
        return(Yc)
    n_train,dim=np.shape(X_train)
    n_test,_=np.shape(X_test)
    # Function Expansion Block
    T_train=np.ones((n_train,dim*(l-1)+1))
    T_test=np.ones((n_test,dim*(l-1)+1))
    for i in range(dim):
        tmp_train_Yc=laguerre(l,X_train[:,i]).T
        tmp_test_Yc=laguerre(l,X_test[:,i]).T
        T_train[:,(i*(l-1)+1):(i*(l-1)+l)]=tmp_train_Yc[:,1:]
        T_test[:,(i*(l-1)+1):(i*(l-1)+l)]=tmp_test_Yc[:,1:]
    W=np.random.rand(dim*(l-1)+1,1)
    old_W=np.random.rand(dim*(l-1)+1,1)
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
        best_W=np.random.rand(dim*(l-1)+1,1)
        t_itr=kwargs['itr']
    except:
        pass
    
    itr=-1
    #Training
    while True:
        itr+=1
        old_W[:]=W[:]
        for i in range(n_train):
            Xi=T_train[i,:].reshape(-1,1)
            MA=np.mean(Xi)
            yp=np.dot(W.T,Xi)+MA
            Y=yp
            error=y_train[i]-Y
            W=W+eta*error*Xi
        epsilon=np.abs(np.linalg.norm(old_W-W,ord='fro'))
        if min_err>np.sum(np.abs(error)):
            best_W=W[:]
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
                break
        except:
            pass
    #Prediction
    MA=np.mean(T_test,axis=1)
    yp=np.dot(W.T,T_test.T)+MA
    Y_hat=yp
    return(Y_hat.flatten(),W)


# # Radial Basis Function Neural Net

# In[ ]:


def RBFN_reg(X_train,y_train,X_test,no_h):
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
    #Step 3: Closed form solution
    pS=np.linalg.pinv(S)
    W=(np.dot(pS,y_train)).T
    #Prediction
    Xi=np.repeat(X_test,no_h,axis=0)
    Xj=np.tile(C,(num2,1))
    S=np.exp(-(np.linalg.norm(Xi-Xj,axis=1)**2)/2).reshape(num2,no_h)
    y_test=np.dot(S,W.T)
    return(y_test)

