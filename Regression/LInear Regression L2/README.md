# LINEAR REGRESSION L2

**SIGNATURE**: 
>linear_ridge_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam)

**DOCSTRING**:
Performs linear regression with L2 regulariser with the given input training data and and predict ( x_test data matrix) result.
**INPUT**:
>fun= The function resembling close to the data given as string
arg_x= Coefficient matrix in ndarray containing each element as string
arg_c= Variable matrix given in ndarray containing each element as string
x_train= Given the training data matrix in ndarray
y_train= Given the training label array in ndarray
x_test= Given the test data matrix in ndarray
lam= Regulariser value.

**PROCESS**:
**Step 1**: Declaring the constant and dependent variables as var of SYMPY using exec()
**Step 2**: Declaring the function using sympify().
**Step 3**: Fetching the training data and feeding the function and forming a list mat_fun to store the values.
**Step 4**:Forming the ‘A’ matrix of **Ax=b** form, where **x** is the coefficient matrix.
**Step 5**: Calculating the coefficient matrix by calculating the pseudo-inverse of A as follows:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Ctheta%3D%5Cleft%20%28%20A%5E%7BT%7D%5Cbullet%20A%20&plus;%20%5Cdelta%5E%7B2%7DI%20%5Cright%20%29%5E%7B-1%7DA%5E%7BT%7D%5Cbullet%20Y)
**RETURN**: 
>y_test: test prediction with shape similar to y_train
C: Coefficient matrix with shape (d×1)