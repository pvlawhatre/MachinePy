# LOGISTIC REGRESSION L1 REGULARISED

**SIGNATURE**: 
>*logistic_lasso_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam,min_itr)*

**DOCSTRING**:
Perform logistic regression with L1 regulariser for binary classification with the given input training data and predict ( x_test data matrix) result for classification.

**INPUT**:
>*fun*= A function from which the data point are assumed to be taken.
*arg_x*= a ndarray of variables name in the function
*arg_c*=a ndarray of coefficients name in the function.
*x_train*= Given the training data matrix in ndarray
*y_train*= Given the training label array in ndarray
*x_test*= Given the test data matrix in ndarray
*eta*= learning rate
*lam*=L1 regulariser
*min_itr*= Minimum number of iterations

**FUNCTIONS INSIDE THE CODE**:
1) *trns_x (xt,fun,arg_x,arg_c)*: Calculate and return the coefficient matrix for the system of ewuations for variable as in *arg_c*.
RETURN: *X*, a ndarray matrix of shape (n,no.of coefficients).  
2) *sigmoid(x_loc)*: Calculates the sigmoid of the given value.  
RETURN: sigmoid value as double.  
3) *p2c(pb)*: Categories the given probabilities into binary classification with probability 0.5 is the boundary for the classification.  
4) _soft_thr(pj,lm,zj)_: Takes the float value of ρ and applying the soft threshold to the given value of ρ with the lamda(lam) value and normalising it with zj.  
RETURN: Normalised C.  

**PROCESS**:
**Step 1**: We choose a random values for the initial guesses for the coefficients from a uniform distribution, C.  
**Step 2**: Declaring the constant and dependent variables as var of SYMPY  
**Step 3**: Declaring function  
**Step 4**: Applying kernel on x. The *X_train* has coefficient matrix for system of equations where variables are the coefficient in *arg_c*. Similarly, for *X_test*.  
**Step 4**: Applying coordinate descent and updating the weight.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D%3D%5Csigma%28xtrain%5Cbullet%20C%29)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%3D-ylog%28%5Cwidehat%7By%7D%29-%281-y%29log%281-%5Cwidehat%7By%7D%29&plus;%5Clambda%5Cleft%20%5C%7C%20w%20%5Cright%20%5C%7C)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?w%3Dx%5Cleft%20%28wx&plus;%28y-%5Cwidehat%7By%7D%29%20%5Cright%20%29)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;applying soft thresholding,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?w%3DS%28%5Crho%2C%5Clambda%29w)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S%28%5Crho%2C%5Clambda%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Cfrac%7B%5Crho&plus;%5Clambda%7D%7B%5Cleft%20%5C%7C%20x%20%5Cright%20%5C%7C%7D%20%26%20%2C%5Crho%3C-%5Clambda%5C%5C%20%5Cfrac%7B%5Crho-%5Clambda%7D%7B%5Cleft%20%5C%7C%20x%20%5Cright%20%5C%7C%7D%26%2C%20%5Crho%3E%5Clambda%20%5Cend%7Bmatrix%7D%5Cright)  

**Step 5**: Prediction using *p2c()* module.  

**RETURN**: 
>*y_lbl*: Classification labels for the given test input.  
*y_test*: test label probabilities  
*C*: Coefficients of the regression function.
