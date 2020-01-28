# LINEAR REGRESSION

**SIGNATURE**: 
>_linear_reg(fun,arg_x,arg_c,x_train,y_train,x_test)_

**DOCSTRING**:  
Performs linear regression.  

**INPUT**:  
>*fun*= The function resembling close to the data given as string  
*arg_x*= Coefficient matrix in ndarray containing each element as string  
*arg_c*= Variable matrix given in ndarray containing each element as string  
*x_train*= Given the training data matrix in ndarray  
*y_train*= Given the training label array in ndarray  
*x_test*= Given the test data matrix in ndarray  

**PROCESS**:  
**Step 1**: Declaring the constant and dependent variables. _Exec()_ function converts both variables and coefficients before substituting values in them. The declaring the function into _sympify()_ function.  
**Step 2**: Fetching the training data and feeding function. _mat_fun_ is the list of fun values for a given point for respective x_train value. Length of _mat_fun_ is the number of points.  
**Step 3**:  Matrix “A” calculation which is from the form Ax=B for many linear equations. _cofmat_ is the list of coefficients.   Here, the x matrix contains the coefficient values and we are intended to calculate the coefficient of the equations.  
**Step 4**: We calculate the pseudo-inverse of A and takes the dot product with the _y_train_ values.  
**Step 5**: We do predictions by substituting the x_test values in the arg_x and then multiplying it with coefficient matrix.  

**RETURN**:  
>*y_test*: the return type ndarray and C is the coefficient matrix.  
