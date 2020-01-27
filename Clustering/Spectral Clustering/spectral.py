import numpy as np

def spectral(X,k,**kwargs):
    num,dim=np.shape(X)
    def similar(xi,xj):
        sig=kwargs['sigma']
        return np.exp(-(np.linalg.norm(xi-xj,axis=1)**2)/2*sig**2)
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
            if a[i,0]<=b[i,0]:
                s[i,0]=1-a[i,0]/b[i,0]
            if a[i,0]>=b[i,0]:
                s[i,0]=b[i,0]/a[i,0]-1
        #avg s
        s_avg=np.mean(s.flatten())
        return(s_avg)
    def kmean_pp(x,k):
        row,col=np.shape(x)
        itr=0
        D=np.empty((row,k))
        Cp=np.empty((k,col))
        #Step 1: Centroid intialisation
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
                D[:,i-1]=(np.linalg.norm(X-c[i-1,:],axis=1))**2
                #nearest centre
                D=np.sort(D,axis=1)
                xt=D[:,0]
                pdf=(xt)/np.sum(xt)
                cdf=np.cumsum(pdf)
                tmp=np.random.uniform(0,1,1)
                tmp2=np.where(cdf>=tmp)[0][0]
                c[i,:]=X[tmp2,:]
            return c
        big_score=0
        for i in range(20):
            C=kpp(x,k)
            while True:
                if row<k:
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
                    C=kpp(x,k)
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
            score=silhouette(x,y,k)
            if score>=big_score:
                big_score=score
                Y=y
        return(Y)
    #step 1: Similarity matrix
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    S=similar(Xi,Xj).reshape(num,num)
    #step 2: Degree matrix
    D=np.identity(num)
    tS=np.sum(S,axis=1)
    D=D*tS
    #Step 3: Laplacian matrix
    if kwargs['laplacian']=='symmetric':
        tD=np.linalg.cholesky(D)
        tD=Dnp.linalg.inv(tD)
        L=np.identity(num)-np.dot(tD,np.dot(S,tD))
    if kwargs['laplacian']=='random-walk':
        L=np.identity(num)-np.dot(np.linalg.inv(D),S)
    if kwargs['laplacian']=='unnormalised':
        L=D-S
    #Step 4: Eignvalues (Following steps reduce dimension, similar points comes closer with this step)
    val,vec=np.linalg.eig(L)
    ind=np.argsort(val)
    val=val[ind]
    vec=vec[:,ind]
    #Step 5: first k eignvalues(smallest)
    V=vec[:,0:k]
    #Step 6: Kmean clsutering
    Y=kmean_pp(V,k)
    return(Y)