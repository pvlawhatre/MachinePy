import numpy as np

def DNE(X,Y,K):
    num,dim=np.shape(X)
    #Step 1: Neighbourhood of the points
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    A=np.linalg.norm(Xi-Xj,axis=1).reshape((num,num))
    A=np.argsort(A,axis=1)
    knn=A[:,1:K+1]
    F=np.zeros((num,num))
    for i in range(num):
        F[i,knn[i,:]]=1
        ind=(Y[i]!=Y)
        multiplier=ind*(-1)
        F[i,:]=F[i,:]*multiplier
    #Step 2: Laplacian of the graph
    D=np.diag(np.sum(F,axis=1))
    L=D-F
    C=np.dot(np.dot(X.T,L),X)
    #EVD
    val,vec=np.linalg.eig(C)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #projection matrix
    p=vec[:,0:K]
    #Trnsformation
    y=np.dot(X,p)
    return (y)