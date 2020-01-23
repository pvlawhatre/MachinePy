import numpy as np

def kernel_PCA(X,k,*args,**kwargs):
    #kernels
    def cosine_similarity(x,y):
        x=np.reshape(x,(np.size(x),1))
        y=np.reshape(y,(np.size(y),1))
        num=np.dot(x.T,y)
        den=(np.linalg.norm(x))*(np.linalg.norm(y))
        return(num/den)
    def linear(x,y):
        return(np.dot(x.T,y))
    def poly(x,y):
        tmp=(np.dot(x.T,y)*kwargs['gamma']+kwargs['c0'])**(kwargs['degree'])
        return(tmp)
    def sigmoid(x,y):
        return( np.tanh(np.dot(x.T,y)*kwargs['gamma']+kwargs['c0']))
    def rbf(x,y):
        tmp=(np.linalg.norm(x-y))**2
        return(np.exp(-tmp*(kwargs['gamma'])))
    #Step 1: kernel formation
    pts,dim=np.shape(X)
    K=np.zeros((pts,pts))
    for i in range(pts):
        for j in range(pts):
            if args[0]=='cosine':
                K[i,j]=cosine_similarity(X[i,:],X[j,:])
            elif args[0]=='linear':
                K[i,j]=linear(X[i,:],X[j,:])
            elif args[0]=='poly':
                K[i,j]=poly(X[i,:],X[j,:])
            elif args[0]=='sigmoid':
                K[i,j]=sigmoid(X[i,:],X[j,:])
            elif args[0]=='rbf':
                K[i,j]=rbf(X[i,:],X[j,:])
            else:
                sys.exit('unidentified kernel')
    #Step 2: Centralised Kernel
    K=K-np.dot((np.identity(pts))/pts,K)-np.dot(K,(np.identity(pts))/pts)+np.dot(np.dot((np.identity(pts))/pts,K),(np.identity(pts))/pts)
    #Step 3:EVD
    val,vec=np.linalg.eig(K)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #k vec and val
    val=val[0:k]
    vec=vec[:,0:k]
    #Step 4:Rescaling eignvectors
    for i in range(k):
        vec[:,i]=vec[:,i]/np.sqrt(val[i]*pts)
    #Projection Matrix
    P=vec
    #Step 5: Projections
    Y=np.dot(K,P)
    return(Y)