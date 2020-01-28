# SEMI-SUPERVISED LEARNING- SELF TRAIN

**SIGNATURE**:   
>_self_train(x_lbl,y_lbl,x_unlbl,K,NN,*arg)_

**DOCSTRING**:  
Perform self-train with the given input training data and predict ( x_test data matrix) result for classification.  

**INPUT**:  
>*x_lbl*= Given the training data matrix in ndarray  
*y_lbl*= Given the training label array in ndarray  
*x_unlbl*= Given the test data matrix in ndarray  
*k*= Top K performer
_NN_= Value of K for KNN
_*args_= 1) **'cityblock'** : Performs cityblock distance. 2) '**minkowsky**': Perfrom Minkowsky distance. 3) '**hamming**': Perform Hamming distance.  
NOTE: If nothing is provided DFAULT Euclidean distance will be performed.  
**kwargs= 'sigma', sigma for similarity matrix in spectral clustring.  

**FUNCTION INSIDE THE CODE**:  
1) _KNN(x_train,y_train,x_test,k,*args)_: Perform KNN for classification with specified X.  
RETURN: *y_test*, label matrix.   

**PROCESS**:  
**Step 1**: Train the model and predict the labels.  
**Step 2**: Top K performers.  
**Step 3**: Append the point in the training dataset.  
**Step 4**: Prediction using KNN for top 'k' values and points.  

**RETURN**:   
>*y_prob*: Test prediction probability with shape similar to y_train.  
*y_cls*: Test prediction with shape similar to y_train.
