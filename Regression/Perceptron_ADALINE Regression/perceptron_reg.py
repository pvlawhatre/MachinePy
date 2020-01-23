import numpy as np

def perceptron_reg(X_train,y_train,X_test,**kwargs):
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