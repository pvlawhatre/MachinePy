import numpy as np

def KNN(x_train,y_train,x_test,k,*args):
    n1,d=np.shape(x_train)
    n2,d=np.shape(x_test)
    y_test=np.zeros(n2)
    def mode(x):
        labels=np.unique(x)
        count_list=np.empty_like(labels)
        for i in range(labels.shape[0]):
            count=0
            for j in range(x.shape[0]):
                if labels[i]==x[j]:
                    count+=1
            count_list[i]=count
        max_ind=np.argmax(count_list)
        return(labels[max_ind],count_list[max_ind])
    def cityblock(u,v):
        d=0
        for i in range(u.shape[0]):
            d=d+np.abs(u[i]-v[i])
        return d
    def minkowsky(u,v):
        dtmp=0
        for i in range(u.shape[0]):
            dtmp=dtmp+np.abs(u[i]-v[i])**2
        d=dtmp**0.5
        return d
    def hamming(u,v):
        c=0
        for i in range(u.shape[0]):
            if u[i]!=v[i]:
                c+=1
        d=c/u.shape[0]
        return d
    def euclidean(u,v):
        dtmp=0
        for i in range(u.shape[0]):
            dtmp=dtmp+np.abs(u[i]-v[i])**2
        d=dtmp**0.5
        return d
    for i in range(n2):
        dist=np.zeros(n1)
        for j in range(n1):
            if args=='cityblock':
                dist[j]=cityblock(x_test[i,:],x_train[j,:])
            elif args=='minkowsky':
                dist[j]=minkowsky(x_test[i,:],x_train[j,:])
            elif args=='hamming':
                dist[j]=hamming(x_test[i,:],x_train[j,:])
            else:
                dist[j]=euclidean(x_test[i,:],x_train[j,:])
        dist=np.argsort(dist)
        tmp=dist[0:k]
        tmp2=y_train[tmp]
        temp3=mode(tmp2)
        y_test[i]=temp3[0]
    return y_test