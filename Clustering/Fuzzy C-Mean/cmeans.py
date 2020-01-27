import numpy as np
import random

def cmeans(x,k,m):
    n,D=np.shape(x)
    def genC(x):
        row,_=np.shape(x)
        c=np.zeros_like(x)
        for i in range(row):
            c[i,:]=jitter(x[i,:])
        np.random.shuffle(c)
        C=c[0:k,:]
        return C

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
    #Step 1: Intialisation
    W=np.random.random((n,k))
    C=genC(x)
    Wp=np.random.random((n,k))
    flag=0
    #Step 2: Update Centroid and weights
    while True:
        Wp[:]=W[:]
        #membership update
        for i in range(n):
            for j in range(k):
                tmp=0
                for p in range(k):
                    num=np.linalg.norm(x[i,:]-C[j,:])
                    den=np.linalg.norm(x[i,:]-C[p,:])
                    res=(num/den)**(2/(m-1))
                    tmp=tmp+res
                W[i,j]=1/tmp
        #Centroid Update
        for i in range(k):
            num=0
            den=0
            for j in range(n):
                num=num+(W[j,i]**m)*x[j,:]
                den=den+(W[j,i]**m)
                res=num/den
            C[i,:]=res
        #Step 3: Stopping Criterion
        dist=np.linalg.norm(W-Wp,ord='fro')
        if dist<0.000000001:
            flag+=1
        else:
            flag=0
        if flag>=100:
            break
        Y=np.argmax(W,axis=1)
    return(Y,W,C)