# SEMI-SUPERVISED LEARNING- CLUSTER AND LABEL

**SIGNATURE**: 
>_cluster_and_label(x_lbl,y_lbl,x_unlbl,*arg,**kwargs)_

**DOCSTRING**:   
Perform cluster and label with the given input training data and predict ( x_test data matrix) result for classification.

**INPUT**:  
>*x_lbl*= Given the training data matrix in ndarray  
*y_lbl*= Given the training label array in ndarray  
*x_unlbl*= Given the test data matrix in ndarray  
_*args_= 1) **'cityblock'** : Performs cityblock distance. 2) '**minkowsky**': Perfrom Minkowsky distance. 3) '**hamming**': Perform Hamming distance.  
NOTE: If nothing is provided DFAULT Euclidean distance will be performed.  
_**kwargs_= 'sigma', sigma for similarity matrix in spectral clustring.  

**FUNCTION INSIDE THE CODE**:  
1) *spctral(X,k)*: Perform spectral clustering with specified X.  
RETURN: *y_test*, ndarray of shape (n,1)  
2) *KNN(x_train,y_train,x_test,k,*args)*: Perform KNN for classification with specified X.  
RETURN: *y_test*, label matrix.  

**PROCESS**:  
**Step 1**: Cluster the data with spectral clustring.  
**Step 2**: For each cluster predict the class with KNN.  
**Step 3**:Make predictions by using KNN on the pseudo-labels.  

**RETURN**:  
>*y_final*: test prediction with shape similar to y_train.
