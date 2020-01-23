import numpy as np

def pca(X,K,*arg):
    #EVD
    def EVD_pca(x,k):
        rw,cl=np.shape(x)
        #step 1: mean calculation
        mean=np.mean(x,axis=0)
        x_t=x-mean
        #step 2: Covariance matrix
        c=(1/(rw-1))*(np.dot(x_t.T,x_t))
        #step 3: evd
        val,vec=np.linalg.eig(c)
        ind=np.argsort(val)[::-1]
        val=val[ind]
        vec=vec[:,ind]
        #step 4: projection matrix
        p=vec[:,0:k]
        #step 5: Transformation
        y=np.dot(x_t,p)
        return y
    #SVD
    def SVD_pca(x,k):
        rw,cl=np.shape(x)
        #step 1: mean calculation
        mean=np.mean(x,axis=0)
        x_t=x-mean
        #step 2: svd
        u,s,v=np.linalg.svd(x_t)
        y=np.dot(u[:,0:k],np.diag(s[0:k]))
        return y
    if arg[0]=='evd':
        y=EVD_pca(X,K)
    elif arg[0]=='svd':
        y=SVD_pca(X,K)
    else: ##if nothing is provided.
        print('unidentified method')
    return y