# SEMI-SUPERVISED LEARNING- PSEUDO LABEL

**SIGNATURE**: 
>_pseudo_label(x_lbl,y_lbl,x_unlbl,k,*arg)_

**DOCSTRING**:  
Perform pseudo labeling with the given input training data and predict ( x_test data matrix) result for classification.  

**INPUT**:  
>*x_lbl*= Given the training data matrix in ndarray  
*y_lbl*= Given the training label array in ndarray  
*x_unlbl*= Given the test data matrix in ndarray  
*k*= 'k' value.  
_*args_= 1) **'cityblock'** : Performs cityblock distance. 2) '**minkowsky**': Perfrom Minkowsky distance. 3) '**hamming**': Perform Hamming distance.  
NOTE: If nothing is provided DFAULT Euclidean distance will be performed.  

**FUNCTION INSIDE THE CODE**:  
1) *KNN(x_train,y_train,x_test,k,*args)*: Performs the KNN for classification.  
RETURN: *y_test*, ndarray of shape (n,1)  

**PROCESS**:  
**Step 1**: Train with labelled data and generate pseudo labels using KNN.  
**Step 2**: Append pseudolabels with training dataset.  
**Step 3**:Make predictions by using KNN on the pseudo-labels.  

**RETURN**:   
>*y_lbl*: test prediction with shape similar to y_train.
