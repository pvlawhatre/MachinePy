# LINEAR REGRESSION L1L2 (ELASTICNET)

**SIGNATURE**: 
>_linear_elasticnet_reg(fun,arg_x,arg_c,x_train,y_train,x_test,lam1,lam2,min_itr)_

**DOCSTRING**:  
Performs linear regression with L1L2 regulariser with the given input training data and predict ( x_test data matrix) result.

**INPUT**:  
>*fun*= The function resembling close to the data given as string  
*arg_x*= Coefficient matrix in ndarray containing each element as string  
*arg_c*= Variable matrix given in ndarray containing each element as string  
*x_train*= Given the training data matrix in ndarray  
*y_train*= Given the training label array in ndarray  
*x_test*= Given the test data matrix in ndarray  
*lam1*= Regulariser value for L1  
*lam2*= Regulariser value for L2  
*min_itr*= A minimum number of times the iteration has to be performed.  

**FUNCTIONS INSIDE THE CODE**:  
1) _trns_X(xt,fun,arg_x,arg_c)_: It takes the value of the given x matrix, variable matrix and coefficient matrix. It adds another column of all values equal to 1.  
RETURN: X, similar to x matrix with extra column of ones.  
2) _soft_thr(pj,lm,zj)_: Takes the float value of ρ and applying the soft threshold to the given value of ρ with the lamda(lam) value and normalising it with zj.  
RETURN: Normalised C.  

**PROCESS**:  
**Step 0**:We choose a random values for the initial guesses for the coefficients from a uniform distribution, C.  
**Step 1**: Declaring the constant and dependent variables as var of SYMPY using exec()  
**Step 2**: Declaring the function using sympify().  
**Step 3**:Applying the kernel on X and make an extra column of ones  
**Step 4**:Apply coordinate descent to minimise the error and updating the coefficient variables with soft threshold. (Note that only difference here is we add L2 regulariser value to the normalising constant value).  
**Step 5**: Predicting the values by applying x_test data into the system of equations.  

**RETURN**:  
>*y_test*: test prediction with shape similar to y_train.  
*C*: Coefficient matrix with shape (d×1)

    
