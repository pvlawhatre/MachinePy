# SEMI-SUPERVISED LEARNING (CO-TRAINING)

**SIGNATURE**:   
>_co_train(x_lbl,y_lbl,x_unlbl,K,NN,*arg)_    

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

**PROCESS**:  
**Step 1**:  Train the model and predict the labels. Two KNN classifiers one each for labelled input data (*x_lbl*) spiltted into half.  
**Step 2**: Top K performers for both the classifiers.  
**Step 3**: Append the points of unlabelled data in the training dataset.  
**Step 4**: Make predicitions from the trained two KNN classifiers. 

**RETURN**:   
>*y_test1*: Test prediction for first half of trianing dataset.
*y_test2*: Test prediction for second half of trianing dataset.