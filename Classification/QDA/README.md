# QUADRATIC DISCRIMINANT ANALYSIS (QDA)

**SIGNATURE**: 
>*QDAClassifier(x_train,y_train,x_test)*

**DOCSTRING**:
Performs LDA for classification to a given data matrix x and predict ( x_test data matrix) result for classification.

**INPUT**:
>*x_train*= Given the training data matrix in ndarray  
*y_train*= Given the training label array in ndarray  
*x_test*= Given the test data matrix in ndarray  

**FUNCTIONS INSIDE THE CODE**:
1) *prior(x_train,y_train)*:Calculates the prior of each label and returns a list of each class respectively.
RETURN: *pik*, a list.  
2) *mean_x(x)*: Takes the training data and return an array of means of all the points in respective class.  
RETURN: *m*, a ndarray of dimension (No. of labels, No. of variables).  
3) *covar_x(x)*: Calculates the covarince and return the matrix.  
RETURN: *c*, a ndarray of dimension ( No. of variables, No. of variables)  

**PROCESS**:  
**Step 1**: Calculates the prior distribution of training data, *pi*  
**Step 2**: Calculating the mean for each class, *mu*  
**Step 3**: Calculate the covariance of the data, *Sk* and its inverse, *Ski*. Where, *Sk* is numpy ndarray with dimension (labels, no. of variables, no. of variables).  
**Step 4**: Predicting the testing value.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?y_%7Bi%2Cj%7D%3Dlog%28p_%7Bi%7D%29&plus;%5Cfrac%7B-1%7D%7B2%7D%28%5Cmu_%7Bi%7D%5CSigma_%7Bi%7D%5E%7B-1%7D%5Cmu_%7Bi%7D%29&plus;x_%7Bi%7D%5CSigma_%7Bi%7D%5E%7B-1%7D%5Cmu_%7Bi%7D&plus;%5Cfrac%7B-1%7D%7B2%7Dx_%7Bi%7D%5CSigma_%7Bi%7D%5E%7B-1%7Dx_%7Bi%7D&plus;%5Cfrac%7B-1%7D%7B2%7Dlog%28%5Cleft%20%7C%20det%28%5CSigma_%7Bi%7D%5E%7B-1%7D%29%20%5Cright%20%7C%29), where *p*= Prior to the distribution, ![](http://latex.codecogs.com/gif.latex?%5CSigma_%7Bi%7D%5E%7B-1%7D)= Covariance matrix of 'i'th label. Classification labels are given by the armax of *y* along each column.  

**RETURN**: 
>*y*: Probability matrix with dimension (labels, No. of test points)  
*y_lbl*: Classification labels for the given test input.
