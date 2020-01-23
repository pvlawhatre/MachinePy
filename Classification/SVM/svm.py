import numpy as np

def SVM(X,Y,x_test,C,*args,**kwargs):
    num,dim=np.shape(X)
    tnum,_=np.shape(x_test)
    #optimisation algorithm
    max_pass=kwargs['max_passes']
    def smo(K,Y):
        Alpha=np.zeros(num)
        B=0
        tol=0.01
        passes=0
        while(max_pass>passes):
            num_changed_alpha=0
            for i in range(num):
                fx=np.sum(Alpha.reshape(-1,1)*Y.reshape(-1,1)*K,axis=0)+B
                fxi=fx[i]
                Ei=fxi-Y[i]
                cnd1=Y[i]*Ei<-tol and Alpha[i]<=C
                cnd2=Y[i]*Ei>tol and Alpha[i]>=0
                if (cnd1==True) or (cnd2==True):
                    while True:
                        j=int(np.random.randint(0,num))
                        if j!=i:
                            break
                    fxj=fx[j]
                    Ej=fxj-Y[j]
                    old_ai=Alpha[i]
                    old_aj=Alpha[j]
                    if Y[i]!=Y[j]:
                        L=max(0,Alpha[j]-Alpha[i])
                        H=min(C,C+Alpha[j]-Alpha[i])
                    else:
                        L=max(0,Alpha[j]+Alpha[i]-C)
                        H=min(C,Alpha[j]+Alpha[i])
                    if L==H:
                        continue
                    eta=2*K[i,j]-K[i,i]-K[j,j]
                    if eta>=0:
                        continue
                    Alpha[j]=Alpha[j]-Y[j]*(Ei-Ej)/eta
                    if Alpha[j]>H:
                        Alpha[j]=H
                    elif Alpha[j]<L:
                        Alpha[j]=L
                    else:
                        pass
                    if np.abs(Alpha[j]-old_aj)<tol:
                        continue
                    Alpha[i]=Alpha[i]+Y[i]*Y[j]*(old_aj-Alpha[j])
                    cnd1=0<Alpha[i] and Alpha[i]<C
                    cnd2=0<Alpha[j] and Alpha[j]<C
                    if cnd1==True and cnd2==False:
                        b1=B-Ei-Y[i]*(Alpha[i]-old_ai)*K[i,i]-Y[j]*(Alpha[j]-old_aj)*K[i,j]
                        B=b1
                    elif cnd1==False and cnd2==True:
                        b2=B-Ej-Y[i]*(Alpha[i]-old_ai)*K[i,j]-Y[j]*(Alpha[j]-old_aj)*K[j,j]
                        B=b2
                    else:
                        b1=B-Ei-Y[i]*(Alpha[i]-old_ai)*K[i,i]-Y[j]*(Alpha[j]-old_aj)*K[i,j]
                        b2=B-Ej-Y[i]*(Alpha[i]-old_ai)*K[i,j]-Y[j]*(Alpha[j]-old_aj)*K[j,j]
                        B=(b1+b2)/2
                    num_changed_alpha=num_changed_alpha+1
            if num_changed_alpha==0:
                passes=passes+1
            else:
                passes=0
        return(Alpha,B)
    #kernels
    def linear(x,y):
        return(np.dot(x,y.T))
    def poly(x,y):
        tmp=(np.dot(x,y.T)*kwargs['gamma']+kwargs['c0'])**(kwargs['degree'])
        return(tmp)
    def rbf(x,y):
        tmp=np.linalg.norm(x-y,axis=1)**2
        return(np.exp(-tmp*(kwargs['gamma'])))
    #Step 1: Transforming input with kernel
    if args[0]=='linear':
        K=linear(X,X).reshape(num,num)
        tK=linear(X,x_test).reshape(num,tnum)
    elif args[0]=='poly':
        K=poly(X,X).reshape(num,num)
        tK=poly(X,x_test).reshape(num,tnum)
    else:
        Xi=np.repeat(X,num,axis=0)
        Xj=np.tile(X,(num,1))
        tXj=np.tile(x_test,(tnum,1))
        K=rbf(Xi,Xj).reshape(num,num)
        tK=rbf(Xi,tXj).reshape(num,tnum)
    #Step 2: Sequential minimal optimization
    Alpha,B=smo(K,Y)
    #Step 3: Predictions
    ty_test=np.sum(Alpha.reshape(-1,1)*Y.reshape(-1,1)*tK,axis=0)+B
    y_test[ty_test>=0]=1
    y_test[ty_test<0]=-1
    return(y_test)