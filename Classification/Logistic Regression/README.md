# LOGISTIC REGRESSION

**SIGNATURE**: 
>logistic_reg(fun,arg_x,arg_c,x_train,y_train,x_test,eta,min_itr)

**DOCSTRING**:
Perform logistic regression for binary classification with the given input training data and predict ( x_test data matrix) result for classification.
**INPUT**:
>fun= A function from which the data point are assumed to be taken.
arg_x= a ndarray of variables name in the function
arg_c=a ndarray of coefficients name in the function.
x_train= Given the training data matrix in ndarray
y_train= Given the training label array in ndarray
x_test= Given the test data matrix in ndarray
eta= learning rate
min_itr= Minimum number of iterations

**FUNCTIONS INSIDE THE CODE**:
1) *trns_x (xt,fun,arg_x,arg_c)*: Calculate and return the coefficient matrix for the system of ewuations for variable as in *arg_c*.
RETURN: *X*, a ndarray matrix of shape (n,no.of coefficients).
2) *sigmoid(x_loc)*: Calculates the sigmoid of the given value.
RETURN: sigmoid value as double.
3) *p2c(pb)*: Categories the given probabilities into binary classification with probability 0.5 is the boundary for the classification.

**PROCESS**:
**Step 1**: Selecting random coefficientn from a uniform distribution.
**Step 2**: Declaring the constant and dependent variables as var of SYMPY
**Step 3**: Declaring function
**Step 4**: Applying kernel on x. The *X_train* has coefficient matrix for system of equations where variables are the coefficient in *arg_c*. Similarly, for *X_test*.
**Step 4**: Applying gradient descent and updating the weight.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D%3D%5Csigma%28xtrain%5Cbullet%20C%29)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%3D-ylog%5Cwidehat%7By%7D-%281-y%29log%281-%5Cwidehat%7By%7D%29)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w%7D%3D%5Cleft%20%28%20%5Cwidehat%7By%7D-y%20%5Cright%20%29%5Cbullet%20x)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?w%3Dw-%5Calpha%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w%7D)
**Step 5**: Prediction using *p2c()* module.

**RETURN**: 
>y_lbl: Classification labels for the given test input.
y_test: test label probabilities
C: Coefficients of the regression function.
