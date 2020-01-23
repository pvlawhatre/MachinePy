import numpy as np

def nadaraya_watson(X_train,Y_train,X_test,**kwargs):
    n_pts1,dim=np.shape(X_train)
    n_pts2,_=np.shape(X_test)
    Y_test=np.zeros(n_pts2)
    def bw(strg,x):
        try:
            n,d=np.shape(x)
        except:
            d=1
            n=np.size(x)
        sgm=np.std(x,axis=0)
        H=np.zeros((d,d))
        if strg=='silverman':
            for i in range(d):
                H[i,i]=(((4/(d+2))**(1/d+4))*(n**(-1/(d+4)))*(sgm[i]))**2
            return H
        elif strg=='scott':
            for i in range(d):
                H[i,i]=((n**(-1/(d+4)))*(sgm[i]))**(2)
            return H
        else:
            print('Unrecognised method for bandwidth selection')
    def kernel(H,x):
        try:
            n,d=np.shape(x)
        except:
            d=1
            n=np.size(x)
        y1=((2*np.pi)**(-d/2))
        y2=((np.linalg.det(H))**(-1/2))
        y31=np.dot(np.transpose(x),np.linalg.inv(H))
        y3=np.exp((-1/2)*np.dot(y31,x))
        y=y1*y2*y3
        return y
    #Step 1: banwidth calculation
    try:
        strg=kwargs['method']
        H=bw(strg,X_train)
    except:
        H=kwargs['BW']
    print('BandWidth: ',H)
    #Step 2: predictions
    for i in range(n_pts2):
        num=0
        den=0
        for j in range(n_pts1):
            num=num+kernel(H,X_test[i,:]-X_train[j,:])*Y_train[j]
            den=den+kernel(H,X_test[i,:]-X_train[j,:])
        Y_test[i]=num/den
    return(Y_test)