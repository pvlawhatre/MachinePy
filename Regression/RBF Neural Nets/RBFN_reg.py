import numpy as np

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