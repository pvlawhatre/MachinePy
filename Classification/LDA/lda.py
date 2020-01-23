import numpy as np

def LDAClassifier(x_train,y_train,x_test):
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
            xc=x[ind,:]
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
    #for i in range(k):
    #    for j in range(pts):
     #       mu_v=np.reshape(mu[i,:],(dim,1))
      #      x_v=np.reshape(x_train[j,:],(dim,1))
       #     t1=np.log(pi[i])
        #    t2=-0.5*(np.dot(np.dot(mu_v.T,Si),mu_v))
         #   t3=np.dot(np.dot(x_v.T,Si),mu_v)
          #  df[i,j]=t1+t2+t3
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