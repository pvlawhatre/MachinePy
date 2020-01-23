# LOCAL RERGRESSION

**SIGNATURE**: 
>LLR(X_trian,Y_train,X_test,tau)

**DOCSTRING**:
Perform local regression with the given input training data and predict ( x_test data matrix) result.
**INPUT**:
>X_train= Given the training data matrix in ndarray
Y_train= Given the training label array in ndarray
X_test= Given the test data matrix in ndarray
tau= Scaling factor for the weight matrix. 

**FUNCTIONS INSIDE THE CODE**:
1) *gauss(strg,x)*: Takes input xi and xj as ndarray and float type repectively
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?f%3Dexp%5Cleft%20%28%20%5Cfrac%7B-%28x_%7Bi%7D-x_%7Bj%7D%29%5E%7B2%7D%7D%7B2%5Ctau%5E%7B2%7D%7D%20%5Cright%20%29)
RETURN: Gaussian function.

**PROCESS**:
**Step 1**: *X* with ones at 1st column
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?X%3D%5Cbegin%7Bbmatrix%7D%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20xtrain%26%20%5Ccdots%26%20%5Cvdots%26%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%5C%5C%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20d&plus;1%29%7D)
**Step 2**: *W*, weight function which is a Gaussian from *gauss()*.
**Step 3**: Closed form solution
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cbeta%3D%28%28%28x%5E%7BT%7Dw%29x%29%5E%7B-1%7D%28x%5E%7BT%7Dw%29%29ytrain)
**Step 4**:Predict the value for the test sets:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?ytest%3D%5Cbegin%7Bbmatrix%7D%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20xtest%26%20%5Ccdots%26%20%5Cvdots%26%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%5C%5C%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20d&plus;1%29%7D%5Cbullet%20%5Cbeta)

**RETURN**: 
>y_test: test prediction with shape similar to y_train.
