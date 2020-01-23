import numpy as np

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