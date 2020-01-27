import numpy as np

def GMM(x,K):
    # Centre intialsation
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
            D[:,i-1]=(np.linalg.norm(X-c[i-1,:],axis=1))**100
            #nearest centre
            D=np.sort(D,axis=1)
            xt=D[:,0]
            pdf=(xt)/np.sum(xt)
            cdf=np.cumsum(pdf)
            tmp=np.random.uniform(0,1,1)
            tmp2=np.where(cdf>=tmp)[0][0]
            c[i,:]=X[tmp2,:]
        return c
    #multivariate gaussian
    def mvg(x,M,H):
        try:
            n,d=np.shape(x)
        except:
            d=1
            n=np.size(x)
        y1=((2*np.pi)**(-d/2))
        y2=((np.linalg.det(H))**(-1/2))
        y31=np.dot(np.transpose(x-M),np.linalg.inv(H))
        y3=np.exp((-1/2)*np.dot(y31,x-M))
        y=y1*y2*y3
        return y
    ####START
    row,col=np.shape(x)
    #Step 1: Intialisation
    #mean
    mu=kpp(x,K)
    #covariance
    sig=np.zeros((K,col,col))
    for i in range(K):
        sig[i,:,:]=np.identity(col)
    #prior
    p=np.ones(K)/K
    #labels
    y=np.zeros((row,K))
    yp=np.zeros((row,K))
    #till cinvergence
    itr=0
    flag=0 
    while True:
        itr+=1
        #Step 2: E-step
        for i in range(row):
            tmp=np.zeros(K)
            tmp2=0
            yp[:]=y[:]
            for j in range(K):
                tmp[j]=mvg(x[i,:],mu[j,:],sig[j,:,:])
                tmp2=tmp2+tmp[j]*p[j]
            for j in range(K):
                y[i,j]=(tmp[j]*p[j])/tmp2
        #Step 3: M-step
        #prior
        p=np.mean(y,axis=0)
        #mean
        tmp=np.dot(y.T,x)/row
        for i in range(K):
            mu[i,:]=tmp[i,:]/p[i]
        #covariance
        for i in range(K):
            xc=x-mu[i,:]
            a=np.reshape(y[:,i],(1,row))
            b=row*p[i]
            a=a/b
            A=np.ones((col,row))
            A=a*A
            A=A.T
            tmp1=xc*A
            sig[i,:,:]=np.dot(xc.T,tmp1)
        #stoppping Criteria
        dist=np.round(np.linalg.norm(y-yp,ord='fro'),decimals=5)
        if dist==0:
            flag=flag+1
        else:
            flag=0
        if flag>=10:
            break
    lbl=np.argsort(y,axis=1)
    lbl=lbl[:,-1]
    return(y,lbl,mu)