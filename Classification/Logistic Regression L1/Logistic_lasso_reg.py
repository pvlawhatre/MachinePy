import numpy as np
from sympy import var,sympify

def logistic_lasso_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam,min_itr):
    #parameters
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    n_train,_=np.shape(x_train)
    n_test,_=np.shape(x_train)
    y_train=np.reshape(y_train,(np.size(y_train),1))
    cofmat=[]
    C=np.random.random((n_ac,1))
    CP=np.zeros_like(C)
    def trns_X(xt,fun,arg_x,arg_c):
        nt,_=np.shape(xt)
        X=np.zeros((nt,n_ac))
        for i in range(nt):
            xs=xt[i,:]
            for j in range(n_ax):
                if j==0:
                    tmpx=f.subs([(arg_x[j],xs[j])])
                else:
                    tmpx=tmpx.subs([(arg_x[j],xs[j])])
            q=sympy.linear_eq_to_matrix((tmpx),cofmat)
            X[i,:]=np.array(q[0]).astype(np.float64)
        return X
    def sigmoid(x_loc):
        x_loc=np.array(x_loc).astype(np.float64)
        return 1/(1+np.exp(-x_loc))
    def p2c(pb):
        cls=np.zeros_like(pb)
        for i in range(n_train):
            if pb[i]>=0.5:
                cls[i]=1
        return cls
    def soft_thr(pj,lm,zj):
        if pj<(-lm):
            return (pj+lm)/zj
        elif pj>lm:
            return (pj-lm)/zj
        else:
            return 0
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    # Step 1:declaring the constant and dependent variables as var of SYMPY
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # Step 2: declaring function
    f=sympify(fun)
    # Step 3: applying kernel on x
    X_train=trns_X(x_train,fun,arg_x,arg_c)
    X_test=trns_X(x_test,fun,arg_x,arg_c)
    # Step 4: Coordinate descent
    itr=0
    while itr<=min_itr:
        itr+=1
        CP[:]=C[:]
        s=sigmoid(np.dot(X_train,C))
        for j in range(n_ac):
            Xj=X_train[:,j].reshape(-1,1)
            Zj=np.linalg.norm(Xj)**2
            tPj=Xj*(y_train-s+C[j,:]*Xj)
            Pj=np.sum(tPj)
            C[j,:]=soft_thr(Pj,lam,Zj)
        dist=np.linalg.norm(C-CP,ord='fro')
        if dist==0:
            flag=flag+1
        else:
            flag=0
        if flag>=10:
            break
    # Step 5: prediction
    y_test=sigmoid(np.dot(X_test,C))
    y_lbl=p2c(y_test)
    print("Number of iterations:", itr)
    return(y_lbl,y_test,C)