import numpy as np

def SSDR_constrained(X,K,**kwargs):
    num,dim=np.shape(X)
    if len(kwargs)==0:
        kwargs['method']='M'
        kwargs['beta']=10
    #determining points in must-link and cannot-link
    try:
        beta=kwargs['beta']
        must_lnk=kwargs['ml']
        ml=np.ones(num)
        for i in range(num):
            ml[i]=np.sum(np.sum(np.isin(must_lnk,X[i,:]),axis=1)==dim)
        nm=np.sum(ml)
        alpha=kwargs['alpha']
        cannot_lnk=kwargs['cl']
        cl=np.ones(num)
        for i in range(num):
            cl[i]=np.sum(np.sum(np.isin(cannot_lnk,X[i,:]),axis=1)==dim)
        nc=np.sum(cl)
    except:
        beta=kwargs['beta']
        must_lnk=kwargs['ml']
        ml=np.ones(num)
        for i in range(num):
            ml[i]=np.sum(np.sum(np.isin(must_lnk,X[i,:]),axis=1)==dim)
        nm=np.sum(ml)
    else:
        pass
    #S matrix
    Xi=np.repeat(X,num,axis=0)
    Xj=np.tile(X,(num,1))
    if kwargs['method']=='M':
        mli=np.repeat(ml.reshape(-1,1),num,axis=0)
        mlj=np.tile(ml.reshape(-1,1),(num,1))
        ml=(mli+mlj).flatten()
        s=np.zeros(num*num)
        s[ml==2]=-beta/nm
        S=s.reshape((num,num))
    elif kwargs['method']=='CM':
        mli=np.repeat(ml.reshape(-1,1),num,axis=0)
        mlj=np.tile(ml.reshape(-1,1),(num,1))
        ml=(mli+mlj).flatten()
        cli=np.repeat(cl.reshape(-1,1),num,axis=0)
        clj=np.tile(cl.reshape(-1,1),(num,1))
        cl=(cli+clj).flatten()
        s=np.zeros(num*num)
        s[ml==2]=-beta/nm
        s[cl==2]=-alpha/nc
        S=s.reshape((num,num))
    else:
        mli=np.repeat(ml.reshape(-1,1),num,axis=0)
        mlj=np.tile(ml.reshape(-1,1),(num,1))
        ml=(mli+mlj).flatten()
        cli=np.repeat(cl.reshape(-1,1),num,axis=0)
        clj=np.tile(cl.reshape(-1,1),(num,1))
        cl=(cli+clj).flatten()
        s=np.ones(num*num)*(1/num**2)
        s[ml==2]=(-beta/nm)+(1/num**2)
        s[cl==2]=(-alpha/nc)+(1/num**2)
        S=s.reshape((num,num))
    #D matrix
    D=np.sum(S,axis=1)
    #Laplacian matrix
    L=D-S
    #A matrix
    A=np.dot(np.dot(X.T,L),X)
    #EVD 
    val,vec=np.linalg.eig(A)
    ind=np.argsort(val)[::-1]
    val=val[ind]
    vec=vec[:,ind]
    #projection matrix
    p=vec[:,0:K]
    #Trnsformation
    y=np.dot(X,p)
    return (y)