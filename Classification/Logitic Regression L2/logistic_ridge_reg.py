import numpy as np
from sympy import var, sympify

def logistic_ridge_reg(fun,arg_x,arg_c,x_train,y_train,x_test,eta,lam,min_itr):
    #parameters
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    n_train,_=np.shape(x_train)
    n_test,_=np.shape(x_train)
    y_train=np.reshape(y_train,(np.size(y_train),1))
    cofmat=[]
    C=np.random.random((n_ac,1))
    CP=np.zeros_like(C)
    def trns_X(xt,arg_x):
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
    X_train=trns_X(x_train,arg_x)
    X_test=trns_X(x_test,arg_x)
    # Step 4: Gradient descent
    itr=0
    while itr<=min_itr:
        itr+=1
        CP[:]=C[:]
        s=sigmoid(np.dot(X_train,C)) 
        tmp=(s-y_train)*X_train  # Gradient of loss function w.r.t. weight without the lambda term
        tmp2=np.sum(tmp,axis=0)
        tmp2=tmp2*eta/n_train
        tmp2=np.reshape(tmp2,(np.size(tmp2),1))
        C=(1-eta*lam/n_train)*CP-tmp2 # weight update
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