import numpy as np

def GP_regressor(x_train,y_train,x_test,sig_y,sig_f,l):
    num,dim=np.shape(x_train)
    num2,dim=np.shape(x_test)
    def kernel(x,xp):
        return (sig_f**2)*np.exp(-(x-xp)**2/(2*l**2))
    #gaussian marginals
    #mu is zero
    #Ky
    Xi=np.repeat(x_train,num,axis=0)
    Xj=np.tile(x_train,(num,1))
    Kt=kernel(Xi,Xj).reshape(num,num)
    Ky=Kt+(sig_y**2)*np.identity(num)
    #K*
    Xi=np.repeat(x_train,num2,axis=0)
    Xj=np.tile(x_test,(num,1))
    Kp=kernel(Xi,Xj).reshape(num,num2)
    #K**
    Xi=np.repeat(x_test,num2,axis=0)
    Xj=np.tile(x_test,(num2,1))
    Kpp=kernel(Xi,Xj).reshape(num2,num2)
    #step 2: gaussian conditionals
    mu=np.dot(np.dot(Kp.T,np.linalg.inv(Ky)),y_train)
    S=Kpp-np.dot(np.dot(Kp.T,np.linalg.inv(Ky)),Kp)
    return(mu,S)