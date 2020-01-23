import numpy as np
from sympy import var,sympify

def linear_reg(fun,arg_x,arg_c,x_train,y_train,x_test):
    # declaring the constant and dependent variables as var of SYMPY
    n_ax=np.size(arg_x)
    n_ac=np.size(arg_c)
    cofmat=[]
    for i in range(n_ax):
        exec("%s = %s" % (arg_x[i],var(arg_x[i])))
    for i in range(n_ac):
        exec("%s = %s" % (arg_c[i],var(arg_c[i])))
    # declaring function
    f=sympify(fun)
    #fetching training data and feeding function
    n_train,n_ax=np.shape(x_train)
    mat_fun=[]
    for i in range(n_train):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_train[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_train[i,j])])
        mat_fun.append(tmp)
    #The "A" Matrix
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    A=np.array(At).astype(np.float64)
    # Pesudo inverse of A
    pA=np.linalg.pinv(A)
    #coefficent prediction for the model
    C=np.dot(pA,y_train)
    #predictions
    n_test,n_ax=np.shape(x_test)
    mat_fun=[]
    for i in range(n_test):
        for j in range(n_ax):
            if j==0:
                tmp=f.subs([(arg_x[j],x_test[i,j])])
            else:
                tmp=tmp.subs([(arg_x[j],x_test[i,j])])
        mat_fun.append(tmp)
    #
    cofmat=[]
    for i in range(n_ac):
        cofmat.append(var(arg_c[i]))
    q=sympy.linear_eq_to_matrix(mat_fun,cofmat)
    At=q[0]
    y_test=np.dot(At,C.T)
    return(y_test,C)