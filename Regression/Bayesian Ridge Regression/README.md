# BAYESIAN RIDGE REGRESSION

**SIGNATURE**: 
>_bayesian_ridge_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam)_

**DOCSTRING**:  
Perform Bayesian ridge regression with the given input training data and predict ( x_test data matrix) result.

**INPUT**:  
>*fun*= The function resembling close to the data given as string  
*arg_x*= Coefficient matrix in ndarray containing each element as string  
*arg_c*= Variable matrix given in ndarray containing each element as string  
*x_train*= Given the training data matrix in ndarray  
*y_train*= Given the training label array in ndarray  
*x_test*= Given the test data matrix in ndarray  
*lam*= Regulariser value  

**PROCESS**:  
**Step 1**: Declaring the constant and dependent variables as var of SYMPY using _exec()_.  
**Step 2**:  Declaring the function using sympify().  
**Step 3**:  Fetching the training data and feeding the function and forming a list mat_fun to store the values.  
**Step 4**:  Forming the ‘A’ matrix of Ax=b form, where x is the coefficient matrix.  
**Step 5**:  Calculating the coefficient matrix by calculating the pseudo-inverse of A as follows:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Ctheta%3D%28A%5E%7BT%7D%5Cbullet%20A&plus;%5Cdelta%5E%7B2%7DI%29%5E%7B-1%7DA%5E%7BT%7D%5Cbullet%20Y)  
**Step 6**:  Feeding the values of x_test data into the system of equation and calculating the y_test values.  
**Step 7**:  Predicting the values of variance and returning it.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Csigma_%7B2%7D%3D%5Cleft%20%5B%20%5Cfrac%7B%28y-A_%7Btrain%7D%5Cbullet%20%5Ctheta%29%5E%7BT%7D%28y-A_%7Btrain%7D%5Cbullet%20%5Ctheta%29%7D%7Bn%7D%20%5Cright%20%5D_%7B%281%5Ctimes%20n%29%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Csigma%3D%5Cbegin%7Bbmatrix%7D%201%5C%5C%201%5C%5C%201%5C%5C%20%5Cvdots%5C%5C%201%5C%5C%20%5Cend%7Bbmatrix%7D%5Cbullet%20%5Csigma_%7B2%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?V%3D%5Cfrac%7B%28A_%7Btrain%7D%5E%7BT%7D%5Cbullet%20A&plus;I%29%7D%7B%5Csigma_%7B2%7D%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Csigma_%7Bi%7D%3D%5Csigma_%7Bi%7D&plus;%28%28A_%7Btest%7D%5Cbullet%20V%29%5Cbullet%20A_%7Btest%7D%5E%7BT%7D%29)  

**RETURN**:  
>*y_test*: test prediction with shape similar to y_train.  
*sigma*:Variance of the likelihood with shape (n_test×1)  
*C*: Coefficient matrix with shape (d×1)  
