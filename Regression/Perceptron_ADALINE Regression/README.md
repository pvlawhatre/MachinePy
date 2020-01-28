# PERCEPTRON/ADALINE REGRESSION

**SIGNATURE**: 
>_perceptron_reg(X_train,y_train,X_test,**kwargs)_

**DOCSTRING**:  
Perform perceptron/ADALINE regression with the given input training data and predict ( x_test data matrix) result.  

**INPUT**:  
>*X_train*= Given the training data matrix in ndarray  
*y_train*= Given the training label array in ndarray  
*X_test*= Given the test data matrix in ndarray  
_**kwargs_= 1) '**eps**': DEFAULT thershold value is 0.0001, stated otherwise. 2) '**stable**': DEFAULT value for the epoch is 10, stated otherwise. 3) '**itr**': Number of iterations to run by. Their is no default value and must be selected prior.  

**PROCESS**:  
**Step 1**: Training of the data.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D%3DW%5E%7BT%7DX&plus;b)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?error%3Dy-%5Chat%7By%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?W%3DW&plus;error%5Ccdot%20X)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?b%3Db&plus;error)  
This process is repeated over and over again till the threshold is reached. The *best_w* and *best_b* is learned.  
**Step 2**: Test the test data and making predictions.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D%3DW%5E%7BT%7D%5Cbullet%20xtest%5E%7BT%7D&plus;b)  

**RETURN**:  
>*y_hat*: test prediction with shape similar to y_train.
*W*: Weight matrix
*b*: Baise
