import numpy as np

def prob_pca(X,K,sigma):
    num,dim=np.shape(X)
    #mean and covariance of X
    M=np.mean(X,axis=0)
    S=np.cov(X.T)
    #EVD of covariance matrix
    val,vec=np.linalg.eig(S)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #MLE 
    Uk=vec[:,0:K]
    Lk=np.diag(val[0:K])
    R=np.identity(K)
    W_ml=np.dot(np.dot(Uk,np.linalg.cholesky(Lk-sigma**2*R),R)
    M_ml=M
    ind=ind[::-1]
    S_ml=np.real(np.mean(val[ind[0:K]])).reshape(-1,1)
    #prediction
    mean=np.dot(np.dot(X-M_ml,W_ml),np.linalg.inv(Lk))
    cov=Lk/sigma**2
    return (mean,cov)