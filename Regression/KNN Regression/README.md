# KNN RERGRESSION

**SIGNATURE**: 
>KNN_reg(x_trian,y_train,x_test,K,P)

**DOCSTRING**:
Perform Gaussian processes regression with the given input training data and predict ( x_test data matrix) result.
**INPUT**:
>x_train= Given the training data matrix in ndarray
y_train= Given the training label array in ndarray
x_test= Given the test data matrix in ndarray
K= K nearest Neighbour int type
P= Order of the norm in string type.

**PROCESS**:
**Step 1**:K-Nearest Neighbour. Order of the norm is user defined and are the same as listed in the 	documentation of linalg.norm as follows:-
None |   Frobenius norm         |      2-norm
-----| ------------------------| --------------
'fro'  |  Frobenius norm        |       --
'nuc'   | nuclear norm            |     --
inf  |    max(sum(abs(x), axis=1))  |     max(abs(x))
-inf    | min(sum(abs(x), axis=1))     |   min(abs(x))
0   |    --                           | sum(x != 0)
1     | max(sum(abs(x), axis=0))       | as below
-1    | min(sum(abs(x), axis=0))    |    as below
2     | 2-norm (largest sing. value)  |    as below
-2    | smallest singular value     |     as below
other  | --                       | sum(abs(x)**ord)**(1./ord)
**Step 2**: Predict the value for the test sets which is the mean of y_train value with k nearest neighbours. 

**RETURN**: 
> y_test: test prediction with shape similar to y_train

    
