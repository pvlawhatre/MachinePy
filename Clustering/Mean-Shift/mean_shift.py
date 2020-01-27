import numpy as np
from random import sample

def mean_shift(x):
    row,col=np.shape(x)
    C=np.empty((row,col))
    def gauss(x):
        c=0.5
        return np.exp(-c*((np.linalg.norm(x))**2))
    def jitter(x):
        v=np.zeros_like(x)
        v[:]=x[:]
        D=np.size(x)
        r_axis=sample(set(range(D)),1)
        r_op=sample([0,1],1)
        r_val=np.random.normal(0.5,1,1)
        if r_op==0:
            c=x[r_axis]+r_val
        else:
            c=x[r_axis]-r_val
        v[r_axis]=c
        return v
    def genC(x):
        row,_=np.shape(x)
        c=np.zeros_like(x)
        for i in range(row):
            c[i,:]=jitter(x[i,:])
        return c
    itr=0
    # Step 1: Initializing random points
    C=genC(x)
    Cp=np.zeros_like(C)
    # Step 2: Calculating the center of gravity
    while True:
        itr=itr+1
        numc,_=np.shape(C)
        for i in range(numc):
            tmp1=0
            tmp2=0
            for j in range(row):
                tmp1=tmp1+(gauss(x[j,:]-C[i,:])*x[j,:])
                tmp2=tmp2+gauss(x[j,:]-C[i,:])
            C[i,:]=tmp1/tmp2
        n_centre,_=np.shape(C)
        dist=np.linalg.norm(C-Cp,ord='fro')
        # STep 3: Repeat until convergence
        if dist<0.000000001:
            break
        Cp[:]=C[:]
    print("No. of iteration=",itr)
    C=np.round_(C, decimals=1)
    C=np.unique(C,axis=0)
    # Step 4: Cluster Formation
    rx,_=np.shape(x)
    rc,_=np.shape(C)
    clust=np.zeros((rx,rc))
    for i in range(rx):
        for j in range(rc):
            clust[i,j]=np.linalg.norm(x[i,:]-C[j,:])
    y=np.zeros(rx)
    y=np.argmin(clust,axis=1)
    return(y,C)