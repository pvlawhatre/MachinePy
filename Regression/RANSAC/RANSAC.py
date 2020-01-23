import numpy as np

def ransac(x_train,y_train,x_test,min_pts,sigma,min_iter):
    num,dim=np.shape(x_train)
    itr=0
    inlier=np.empty((num,min_iter))
    while(itr<min_iter):
        #Step 1: Randomly select min points for regression
        ind=random.sample(range(0,num),min_pts)
        xin=x_train[ind,:]
        yin=y_train[ind,:]
        #step 2: Perform regression
        tmp=np.ones((min_pts,1))
        Xin=np.hstack((tmp,xin))
        iX=np.linalg.pinv(Xin)
        Cin=np.dot(iX,yin)
        #Step 3: inliers
        X_train=np.hstack((np.ones((num,1)),x_train))
        Din=np.abs((np.dot(X_train,Cin)-y_train)/np.sqrt(1+np.linalg.norm(Cin)**2))
        cnd=(Din<=sigma) # Thresholding
        inlier[:,itr]=cnd.flatten()
        itr=itr+1
    #Step 4: model selection
    itrid=np.argmax(np.sum(inlier,axis=0))
    ind=inlier[:,itrid]
    ind=(ind==1)
    num=np.sum(ind)
    xin=x_train[ind,:]
    yin=y_train[ind,:]
    tmp=np.ones((num,1))
    Xin=np.hstack((tmp,xin))
    iX=np.linalg.pinv(Xin)
    Cin=np.dot(iX,yin)
    #Step 5: Prediction 
    num,_=np.shape(x_test)
    tmp=np.ones((num,1))
    Xt=np.hstack((tmp,x_test))
    y_test=np.dot(Xt,Cin)
    return(xin,yin,Cin,y_test)