import numpy as np

def LLR(X_train,Y_train,X_test,tau):
    num1,dim=np.shape(X_train)
    num2,_=np.shape(X_test)
    Y_test=np.zeros(num2)
    def gauss(xi,xj):
        return np.exp(-(xi-xj)**2/(2*tau**2))
    for i in range(num2):
        #Step 1: X with ones at 1st column
        dX=X_train
        col1=np.ones((num1,1))
        X=np.hstack((col1,dX))
        #Step 2: W
        w=gauss(X_train,X_test[i,:]).flatten()
        W=np.diag(w)
        #Step 3: closed form solution
        p1=np.dot(X.T,W)
        p2=np.linalg.inv(np.dot(p1,X))
        p3=np.dot(p2,p1)
        beta=np.dot(p3,Y_train.reshape(-1,1))
        #Step4: Prediction
        xt=X_test[i,:].reshape(1,-1)
        xt=np.hstack((np.array([[1]]),xt))
        Y_test[i]=np.dot(xt,beta)
    return(Y_test)