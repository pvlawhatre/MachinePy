import numpy as np

def HCA(x,k):
    # Ward method
    def sim(x,y):
        nx=np.shape(x)
        nx=int(nx[0])
        ny=np.shape(y)
        ny=int(ny[0])
        dist=0
        itr=0
        for i in range(nx):
            for j in range(ny):
                itr=itr+1
                d=(np.linalg.norm(x[i]-y[j]))**2
                dist=dist+d
        ward_dist=dist/itr
        return(ward_dist)
    def low_mat(D):
        n,_=np.shape(D)
        v=np.arange(1,n,1)
        itr=0
        for i in v:
            for j in range(n):
                itr=itr+1
                if itr==1:
                    min_val=D[i,j]
                    min_i=i
                    min_j=j
                else:
                    if j<i and D[i,j]<min_val:
                        min_val=D[i,j]
                        min_i=i
                        min_j=j
        return(min_i,min_j)

    n,_=np.shape(x)
    y=np.array(range(n))
    # Step 1: N individual clusters
    cn=n
    itr=-1
    while cn!=k:
        itr=itr+1
        nt=np.size(np.unique(y))
        D=np.zeros((nt,nt))
        #Step 2: Calculate the proximity of individual points 
        # and consider all the data points as individual clusters.
        for i in range(nt):
            for j in range(nt):
                cnd1=(y==i*np.ones_like(y))
                cnd2=(y==j*np.ones_like(y))
                cls1=np.array(x[cnd1,:])
                cls2=np.array(x[cnd2,:])
                D[i,j]=sim(cls1,cls2)
        mi,mj=low_mat(D)
        # Step 3: Similar clusters are merged together and formed as a single cluster
        min_ij=np.min(np.array([mi,mj]))
        max_ij=np.max(np.array([mi,mj]))
        cnd=(max_ij*np.ones_like(y)==y)
        y[cnd]=min_ij
        cnd=(max_ij*np.ones_like(y)<y)
        y[cnd]=y[cnd]-1
        cn=np.size(np.unique(y))
        # Step 4: Again calculate the proximity of new clusters
        # and merge the similar clusters to form new clusters.
    print("iterations=",itr)
    return y