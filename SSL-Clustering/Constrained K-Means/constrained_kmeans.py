import numpy as np

def constrained_kmeans(x_lbl,y_lbl,x_unlbl):
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
    while True:
        Xi=np.repeat(x_unlbl,K,axis=0)
        Xj=np.tile(C,(n2,1))
        S=np.linalg.norm(Xi-Xj,axis=1).reshape(n2,K)
        tmp=np.argsort(S,axis=1)
        y_unlbl=tmp[:,0]
        dist=np.linalg.norm(y_lbl)
        if np.abs(dist-pdist)<=0.0001:
            flag+=1
        else:
            flag=0
        if flag==3:
            break
        pdist=dist
        y_tot[n1:(n1+n2)]=y_unlbl
        for i in range(K):
            tx=x_tot[i==np.array(y_tot),:]
            C[i,:]=np.mean(tx,axis=0)
        pdist=dist
    return(y_unlbl)