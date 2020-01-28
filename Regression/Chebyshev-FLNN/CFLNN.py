import numpy as np

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