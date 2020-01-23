import numpy as np

def self_train(x_lbl,y_lbl,x_unlbl,K,NN,*arg):
    test=x_unlbl
    y_lbl=y_lbl.reshape(-1,1)
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
    itr=0
    while True:
        itr+=1
        print(itr)
        #Step 1: train the model and predict the labels
        y_prob,y_cls=KNN(x_lbl,x_unlbl,y_lbl.flatten(),NN)
        #Step 2: top K performers
        tmp_pts=np.argsort(np.max(y_prob,axis=0))[::-1]
        tmp_cls=y_cls[tmp_pts]
        tmp_pt=tmp_pts[0:K]
        tmp_cl=tmp_cls[0:K]
        #Step 3: Append the point in the training datset
        xt=x_unlbl[tmp_pt,:]s
        x_unlbl=np.delete(x_unlbl,tmp_pt,0)
        x_lbl=np.append(x_lbl,xt,axis=0)
        y_lbl=np.append(y_lbl,tmp_cl.reshape(-1,1),axis=0)
        if np.size(x_unlbl)==0:
            break
    #Step 4: predict
    y_prob,y_cls=KNN(x_lbl,test,y_lbl.flatten(),NN)
    return(y_prob,y_cls)