import numpy as np

def seed_kmeans(x_lbl,y_lbl,x_unlbl):
    n1,d=np.shape(x_lbl)
    n2,_=np.shape(x_unlbl)
    y_unlbl=np.zeros(n2)
    K=int(np.size(x_lbl)/d)
    C=np.zeros((K,d))
    #Step 1: centre initialisation
    for i in range(K):
        tx=x_lbl[i==y_lbl,:]
        C[i,:]=np.mean(tx,axis=0)
    #Step 2:Points assignment
    y_tmp=np.zeros_like(y_unlbl)
    flag=0
    x_tot=np.vstack((x_lbl,x_unlbl))
    y_tot=np.hstack((y_lbl,y_unlbl))
    pdist=0
    X=np.vstack((x_lbl,x_unlbl))
    Y=np.hstack((y_lbl,y_unlbl))
    while True:
        Xi=np.repeat(X,K,axis=0)
        Xj=np.tile(C,(n2+n1,1))
        S=np.linalg.norm(Xi-Xj,axis=1).reshape(n2+n1,K)
        tmp=np.argsort(S,axis=1)
        Y=tmp[:,0]
        dist=np.linalg.norm(Y)
        if np.abs(dist-pdist)<=0.0001:
            flag+=1
        else:
            flag=0
        if flag==3:
            break
        pdist=dist
        for i in range(K):
            tx=X[i==Y,:]
            C[i,:]=np.mean(tx,axis=0)
        pdist=dist
    return(X,Y)