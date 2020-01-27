import numpy as np

def co_train(x_lbl,y_lbl,x_unlbl,K,NN,*arg):
    test=np.zeros_like(x_unlbl)
    test[:]=x_unlbl[:]
    num_lbl,dim=np.shape(x_lbl)
    if num_lbl%2!=0:
        split=(num_lbl+1)/2
    else:
        split=num_lbl/2
    split=int(split)
    L1=x_lbl[0:split-1,:]
    Y1=y_lbl[0:split-1]
    L2=x_lbl[split:num_lbl,:]
    Y2=y_lbl[split:num_lbl]
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
                cnt[i]=temp3[1]
            return (cnt,y_test)
    itr=0
    while True:
        itr+=1
        print(itr)
         #Step 1: train the model and predict the labels
        y_prob1,y_cls1=KNN(L1,x_unlbl,Y1.flatten(),NN)
        y_prob2,y_cls2=KNN(L2,x_unlbl,Y2.flatten(),NN)
        #Step 2: top K performers
        tmp_pts1=np.argsort(np.max(y_prob1,axis=0))[::-1]
        tmp_pts2=np.argsort(np.max(y_prob2,axis=0))[::-1]
        tmp_cls1=y_cls1[tmp_pts1]
        tmp_cls2=y_cls2[tmp_pts2]
        tmp_pt1=tmp_pts1[0:K]
        tmp_pt2=tmp_pts2[0:K]
        tmp_cl1=tmp_cls1[0:K]
        tmp_cl2=tmp_cls2[0:K]
        #Step 3: Append the points in the training dataset
        xt1=x_unlbl[tmp_pt1,:]
        xt2=x_unlbl[tmp_pt2,:]
        tmp_pt=np.unique(np.hstack((tmp_pt1,tmp_pt2)))
        x_unlbl=np.delete(x_unlbl,tmp_pt,0)
        L1=np.append(L1,xt2,axis=0)
        L2=np.append(L2,xt1,axis=0)
        Y1=np.append(Y1,tmp_cl2,axis=0)
        Y2=np.append(Y2,tmp_cl1,axis=0)
        if np.size(x_unlbl)==0:
            break
    #Step 4: predict
    _,y_test1=KNN(L1,test,Y1.flatten(),NN)
    _,y_test2=KNN(L2,test,Y2.flatten(),NN)
    return(y_test1,y_test2)