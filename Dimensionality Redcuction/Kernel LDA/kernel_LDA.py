import numpy as np
def kernel_LDA(X,Y,k,*args,**kwargs):
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
    #coding matrix
    def coding_mat(y):
        ncls=np.size(np.unique(y))
        _,ccls=np.unique(y,return_counts=True)
        z=np.zeros((pts,ncls))
        for i in range(pts):
            for j in range(ncls):
                if y[i]==j:
                    z[i,j]=np.sqrt(pts/ccls[j])-np.sqrt(ccls[j]/pts)
                else:
                    z[i,j]=-np.sqrt(ccls[j]/pts)
        return(z)
                
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
    #Step 2: Centering kernel
    J=np.identity(pts)-np.ones((pts,pts))/pts
    Kc=np.dot(np.dot(J,K),J)
    #Step 3: Total scatter matrix
    St=np.dot(Kc,Kc)
    #Step 4: Between class scatter matrix and coding matrix
    Z=coding_mat(Y)
    p1=np.dot(Kc,Z)
    p2=np.dot(Z.T,Kc)
    Sb=np.dot(p1,p2)/pts
    #Step 5: Within class scatter matrix
    Sw=St-Sb
    #Step 6: Ratio 
    sw=np.linalg.inv(Sw)
    S=np.dot(sw,Sb)
    #Step 7: EVD
    val,vec=np.linalg.eig(S)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #step 8: projection matrix
    p=vec[:,0:k]
    #step 9: Transformation
    x_trans=np.dot(K,p)
    return x_trans