import numpy as np

def QDAClassifier(x_train,y_train,x_test):
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
    # for i in range(k):
    #     for j in range(pts):
    #         mu_v=np.reshape(mu[i,:],(dim,1))
    #         x_v=np.reshape(x_train[j,:],(dim,1))
    #         t1=np.log(pi[i])
    #         t2=-0.5*(np.dot(np.dot(mu_v.T,Ski[i,:,:]),mu_v))
    #         t3=np.dot(np.dot(x_v.T,Ski[i,:,:]),mu_v)
    #         t4=-0.5*np.dot(np.dot(x_v.T,Ski[i,:,:]),x_v)
    #         t5=-0.5*(np.log(np.abs(np.linalg.det(Ski[i,:,:]))))
    #         df[i,j]=t1+t2+t3+t4+t5
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