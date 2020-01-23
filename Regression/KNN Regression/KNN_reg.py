import numpy as np

def KNN_reg(x_train,y_train,x_test,K,P):
    num1,dim=np.shape(x_train)
    num2,_=np.shape(x_test)
    y_test=np.zeros(num2)
    #Step 1: K nearest neighbour
    Xi=np.repeat(x_test,num1,axis=0)
    Xj=np.tile(x_train,(num2,1))
    S=np.linalg.norm(Xi-Xj,ord=P,axis=1).reshape(num2,num1)
    S=np.argsort(S,axis=1)
    D=S[:,0:K-1]
    #Step 2: Prediction
    for i in range(num2):
        y_test[i]=np.mean(y_train[D[i,:]])
    return y_test