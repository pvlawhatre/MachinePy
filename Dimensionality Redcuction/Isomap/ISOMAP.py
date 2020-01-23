import numpy as np

def isomap(X,n_k,k): ##input:(data matrix,K for KNN,k is the desired dimension)
    num,dim=np.shape(X)
    def nonisolated_G(n_k):
        #Centering matrix
        def centring_mat(num):
            a1=np.identity(num)
            a2=np.ones((num,num))/num
            return (a1-a2)
        #Geodesic distance(Dijkstra Algorithm)
        def geodesic(Net,source):
            size,_=np.shape(Net)
            dist=np.zeros(size)
            #Initialisation of the graph
            dist[:]=np.inf
            dist[source]=0
            visit=np.arange(0,size,1)
            visit=visit.tolist()
            visited=[]
            seed=source
            for i in range(size):
                if Net[seed,i]!=np.inf:
                    dist[i]=Net[seed,i]
            visited.append(seed)
            visit.pop(visit.index(seed))
            #loop 
            while True:
                stg_dist=[dist[i] for i in visit]
                stg_ind=[i for i in visit]
                tmp=np.argsort(stg_dist)
                seed=stg_ind[tmp[0]]
                #Relaxation
                for i in range(size):
                    if dist[seed]+Net[seed,i]<dist[i] and i not in visited:
                        dist[i]=dist[seed]+Net[seed,i]
                visited.append(seed)
                visit.pop(visit.index(seed))
                if visit==[]:
                    break     
            return dist
        def L2norm():
            L2=np.zeros((num,num))
            for i in range(num):
                for j in range(num):
                    L2[i,j]=np.linalg.norm(X[i,:]-X[j,:])
            return L2
        def graph(L2):
            G=np.empty((num,num))
            G[:]=np.inf
            for i in range(num):
                gtmp=L2[i,:]
                s_ind=np.argsort(gtmp)
                s_val=np.sort(gtmp)
                for j in range(1,n_k+1):
                    G[i,s_ind[j]]=s_val[j]
            return G
        #L2 distance matrix
        L2=L2norm()
        #network
        N=graph(L2)
        #Step 1:Geodesic distance
        D=np.zeros((num,num))
        for i in range(num):
            D[i,:]=geodesic(N,i)
        #Step 2: Gower transformation
        #step 2.1: -0.5*d^2
        tmp=np.square(D)
        tmp=(-0.5)*tmp
        #step 2.2: centering (double)
        C=centring_mat(num)
        G=np.dot(np.dot(C,tmp),C)
        return G
    # Step 3: EVD decomposition
    while True:
        G=nonisolated_G(n_k)
        flg=0
        try:
            val,vec=np.linalg.eig(G)
        except:
            flg=1
        if flg==1:
            n_k=n_k+1
            print('Increasing the neighbourhood to ',n_k)
            continue
        else:
            break
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