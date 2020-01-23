import numpy as np

def LDA(x,y,k):
    n,d=np.shape(x)
    #no of cls
    cls=np.unique(y)
    nc=np.size(cls)
    #class mean intilisation
    mc=np.zeros((nc,d))
    #Step 1: Mean
    #global mean
    m=np.mean(x,axis=0)
    m=np.reshape(m,(d,1))
    #class mean
    for i in range(nc):
        ind=(y==cls[i])
        xt=x[ind,:]
        mc[i,:]=np.mean(xt,axis=0)
    #Step 2: scatter matrix
    #step 2.1: Within class
    Sw=np.zeros((d,d))
    npt=[]
    for i in range(nc):
        ind=(y==cls[i])
        xt=x[ind,:]
        mt=np.reshape(mc[i,:],(d,1))
        #number of points in each class
        npts,_=np.shape(xt)
        npt.append(npts)
        stmp=np.zeros((d,d))
        for j in range(npts):
            xs=np.reshape(xt[j,:],(int(d),1))
            dst=xs-mt
            stmp=stmp+np.dot(dst,dst.T)
        Sw=Sw+stmp
    #step 2.2: Between class
    Sb=np.zeros((d,d))
    for i in range(nc):
        mt=np.reshape(mc[i,:],(d,1))
        dst=mt-m
        Sb=Sb+np.dot(dst,dst.T)*npt[i]
    #Step 3: Ratio 
    sw=np.linalg.inv(Sw)
    S=np.dot(sw,Sb)
    #Step 4: EVD
    val,vec=np.linalg.eig(S)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #step 4: projection matrix
    p=vec[:,0:k]
    #step 5: Trnsformation
    x_trans=np.dot(x,p)
    return x_trans