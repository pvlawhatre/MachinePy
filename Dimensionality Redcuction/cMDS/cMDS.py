import numpy as np

def cMDS(x,k):
    n,d=np.shape(x)
    D=np.empty((n,n))
    def centering_mat(num):
        a1=np.identity(num)
        a2=np.ones((num,num))/num
        return (a1-a2)
    #step 1: Proximity matrix/ Dissimilarity matrix
    for i in range(n):
        for j in range(n):
            D[i,j]=np.linalg.norm(x[i,:]-x[j,:])
    #Step 2: Gower transformation
    #step 2.1: -0.5*d^2
    tmp=np.square(D)
    tmp=(-0.5)*tmp
    #step 2.2: centring (double)
    H=centring_mat(num)
    G=np.dot(np.dot(H,tmp),H)
    # Step 3: EVD decomposition 
    val,vec=np.linalg.eig(G)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #step 4: projection matrix
    pval=np.sqrt(val[0:k])
    pvec=vec[:,0:k]
    for i in range(k):
        pvec[:,i]=pvec[:,i]*pval[i]
    #Step 5: Principle coordinates/transformation of X
    y=pvec
    return y