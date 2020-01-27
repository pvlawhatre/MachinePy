import numpy as np
from random import sample

def kmean(x,k):  ##kmean(data_matrix, K)
    row,col=np.shape(x)
    itr=0
    D=np.empty((row,k)) #Distance Matrix
    Cp=np.empty((k,col)) # Center Point Matrix

    # Funtion for Center Initialisation
    def cenini(x,k):
        _,col=np.shape(x)
        C=np.empty((k,col))
        min=np.min(x,axis=0)
        max=np.max(x,axis=0)
        #Selecting K random points for Center Initialisation
        for i in range(col):
            C[:,i]=np.array(sample(set(np.linspace(min[i],max[i],100)),k))
        return C
    # Silhoutte score to check the accuracy
    def silhouette(X,Y,K):
        num,dim=np.shape(X)
        a=np.zeros((num,1))
        b=np.zeros((num,1))
        s=np.zeros((num,1))
        def dist(pt,xt):
            num,dim=np.shape(xt)
            pt=pt.reshape(1,dim)
            Xi=np.repeat(pt,num,axis=0)
            Xj=xt
            S=np.linalg.norm(Xi-Xj,axis=1)
            return np.sum(S)/(num)
        for i in range(num):
            p=X[i,:]
            #a 
            ind_i=(Y[i]==Y)
            ind_i[i]=False
            xt=X[ind_i,:]
            a[i,0]=dist(p,xt)
            #b
            ttmp=[]
            for j in range(K):
                if j==Y[i]:
                    continue
                xt=X[j==Y,:]
                tmp=dist(p,xt)
                ttmp.append(tmp)
            b[i,0]=np.min(np.array(ttmp))
            #s
            if a[i,0]==b[i,0]:
                s[i,0]=0
            if a[i,0]<b[i,0]:
                s[i,0]=1-a[i,0]/b[i,0]
            if a[i,0]>b[i,0]:
                s[i,0]=b[i,0]/a[i,0]-1
        #avg s
        s_avg=np.mean(s.flatten())
        return(s_avg)
    big_score=0
    for i in range(50):
        iter=0
        #Step 1: Centroid intialisation
        C=cenini(x,k)
        while True:
            if row<k: ## INVALID ENTRY
                print('Number of centroids exceeds the number of points')
                break
            itr=itr+1
            #Step 2: Distance matrix
            for i in range(row):
                for j in range(k):
                    D[i,j]=np.linalg.norm(x[i,:]-C[j,:])
            #Step 3: Cluster assignment
            y=np.argmin(D,axis=1)
            if np.size(np.unique(y))!=k:
                itr=0
                C=cenini(x,k)
                continue
            #Step 4: Update Centroid
            Cp[:]=C[:]
            for i in range(k):
                cnd=(i*np.ones_like(y)==y)
                xg=x[cnd,:]
                C[i,:]=np.mean(xg,axis=0)
            #Step 5: Stopping criterion
            distm=np.linalg.norm(C-Cp,ord='fro')
            if distm==0:
                break
            iter+=1
        score=silhouette(x,y,k) #Slihoutte score to check the accuracy.
        if score>big_score:
            big_score=score
            Y=y
            c=C
    return(Y,c) #Return label vector Y and centroid vector c
