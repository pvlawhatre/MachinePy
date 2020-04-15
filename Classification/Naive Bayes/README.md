# NAIVE BAYES

**SIGNATURE**: 
>*gaussian_NB(x_train,y_train,x_test)*

**DOCSTRING**:
Performs Naive Bayes for classification to a given data matrix x and predict ( x_test data matrix) result for classification.

**INPUT**:  
>*x_train*= Given the training data matrix in ndarray  
*y_train*= Given the training label array in ndarray  
*x_test*= Given the test data matrix in ndarray  

**FUNCTIONS INSIDE THE CODE**:  
1) *normal(x,p1,p2)*:Returns the normal distribution for the given mean and standard deviation.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?p%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%5Csigma%7Dexp%5Cleft%20%28%20%5Cfrac%7B-%28x-%5Cmu%29%5E%7B2%7D%7D%7B2%5Csigma%5E%7B2%7D%7D%20%5Cright%20%29)  
RETURN: *p*, a ndarray.  

**PROCESS**:  
**Step 1**: Calculates the prior distribution of training data, *pi*  
**Step 2**: Calculating the likelihood parameters (mean and standard deviation)  for each class, *mu* and *sig*.  
**Step 3**: Calculate the posterior.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?y%3Dp%5Ccdot%20Nor%28xtest%3B%5Cmu%2C%5Csigma%29)  
**Step 4**: Predicting the testing value. Classification labels are given by the argmax of *y* along each column.  

**RETURN**:  
>*y_lbl*: Classification labels for the given test input.  
*y_tmp2*: Index matrix of respective test vaalue in label.  
