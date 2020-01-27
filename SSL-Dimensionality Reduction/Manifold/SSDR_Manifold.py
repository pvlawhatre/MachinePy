import numpy as np

def SSDR_manifold(Xl,Yl,Xu,K,gamma):
    num_l,dim=np.shape(Xl)
    num_u,dim=np.shape(Xu)
    num=num_l+num_u
    X=np.vstack((Xl,Xu))
    def gauss(xi,xj,si,sj):
        return np.exp(-np.linalg.norm(xi-xj,axis=1)**2/(si*sj))
    #Step 1: labelled cost function
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    A=np.linalg.norm(Xi-Xj,axis=1).reshape((num,num))
    iA=np.argsort(A,axis=1)
    knn=iA[:,1:K+1]
    C_l=np.zeros((num,num))
    for i in range(num_l):
        #points in the same cluster
        C_l[i,knn[i,:]]=1
        ind=(Yl[i]!=Yl)
        #points in differnet cluster
        multiplier=ind*(-1)
        C_l[i,0:num_l]=C_l[i,0:num_l]*multiplier
        #unlabbeled points
        ind2=(knn[i,:]>=num_l)
        C_l[i,knn[i,:]]=0
    #Step 2: unlebelled cost function
    sigma=np.zeros(num)
    iA=np.sort(A,axis=1)
    sigma=iA[:,K]
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    Si=np.repeat(sigma,num,axis=0)
    Sj=np.tile(sigma,(num,1)).flatten()
    C_u=gauss(Xi,Xj,Si,Sj).reshape((num,num))
    C=C_l+gamma*C_u
    #Step 3: Laplacian of the graph
    D=np.diag(np.sum(C,axis=1))
    L=D-C
    T=np.dot(np.dot(X.T,L),X)
    #EVD(generalised eign value problem)
    val,vec=np.linalg.eig(T)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #projection matrix
    p=vec[:,0:K]
    #Trnsformation
    y=np.dot(X,p)
    return y