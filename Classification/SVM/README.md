# SUPPORT VECTOR MACHINE (SVM)

**SIGNATURE**: 
>*SVM(X,Y,x_test,C,*args,**kwargs)*

**DOCSTRING**:
Performs Support Vector Machine for classification to a given data matrix x and predict ( x_test data matrix) result for classification.

**INPUT**:
>*X*= Given the training data matrix in ndarray  
*Y*= Given the training label array in ndarray  
*x_test*= Given the test data matrix in ndarray  
*C*= SVM Hyperparameter.  
_*arg_: Arg have two input types- LINEAR, POLY and RBF (Default) for linear function, polynomial function and radial basis function respectively. It must be given as list or a string. **’linear’** , **’poly’** and **‘rbf’** respectively.
**kwargs: Kwarg value for maximum passes as **'max_passes'**. Kwarg values for the respective arg functions are below:-

Funtions | Input 
--------- | ------------
‘linear’ | NOTHING LEAVE IT EMPTY
‘poly’ | gamma=<float32t>, c0=<float32>,degree=<int>
‘rbf’ | gamma=<float32>

**FUNCTIONS INSIDE THE CODE**:  
1) _linear(x,y)_: When *arg is ‘linear’.  
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3Dx_%7B1%7Dx_%7B2%7D%5E%7BT%7D)  
RETURN: The kernel _K_ of dimension same as of x.  
2) _poly(x,y)_:	When *arg is ‘poly’.  
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3D%28%28x_%7B1%7Dx_%7B2%7D%5E%7BT%7D%29%5Cgamma&plus;c_%7B0%7D%29%5E%7Bdeg%7D)  
RETURN: The kernel _K_ of dimension same as of x.  
3) _rbf(x,y)_:	When *arg is ‘rbf’.  
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3Dexp%28-%5Cleft%20%7C%20x_%7B1%7D-x_%7B2%7D%20%5Cright%20%7C%5E%7B2%7D%5Cgamma%29)  
RETURN: The kernel _K_ of dimension same as of x.  
6) *smo(K,Y)*: Sequantial minimal optimization function.  
RETURN: Alpha,B= Lagragian multiplier and bias respectively.  

**PROCESS**:  
**Step 1**: Transforming input with kernel,*k*,*tk*.  
**Step 2**: Do sequential minimal optimization, *Alpha*,*B*.  
The algorithm is as follows:  
        1) Find a Lagrange multiplier ![](http://latex.codecogs.com/gif.latex?%5Calpha_%7B1%7D) that violates the Karush–Kuhn–Tucker (KKT) conditions for the optimization problem.  
        2) Pick a second multiplier ![](http://latex.codecogs.com/gif.latex?%5Calpha_%7B2%7D) and optimize the pair ![](http://latex.codecogs.com/gif.latex?%28%5Calpha_%7B1%7D%2C%5Calpha_%7B2%7D%29).  
        3) Repeat steps 1 and 2 until convergence.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?f%28x%29%3D%5Csum%20%5Calpha%5Cleft%20%28%20yK%28x_%7Bi%7Dx_%7Bj%7D%29%20%5Cright%20%29&plus;B)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?E_%7Bi%7D%3Df%28x_%7Bi%7D%29-y_%7Bi%7D),&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?E_%7Bj%7D%3Df%28x_%7Bj%7D%29-y_%7Bj%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%3D%5Cbegin%7Bcases%7D%20max%280%2C%5Calpha_%7Bj%7D-%5Calpha_%7Bi%7D%29%20%26%20%5Ctext%7B%20if%20%7D%20y%5E%7Bi%7D%5Cneq%20y%5E%7Bj%7D%20%5C%5C%20max%280%2C%5Calpha_%7Bi%7D&plus;%5Calpha_%7Bj%7D-C%29%20%26%20%5Ctext%7B%20if%20%7D%20y%5E%7Bi%7D%3Dy%5E%7Bj%7D%20%5Cend%7Bcases%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?H%3D%5Cbegin%7Bcases%7D%20min%28C%2CC&plus;%5Calpha_%7Bj%7D-%5Calpha_%7Bi%7D%29%20%26%20%5Ctext%7B%20if%20%7D%20y%5E%7Bi%7D%5Cneq%20y%5E%7Bj%7D%20%5C%5C%20min%28C%2C%5Calpha_%7Bi%7D&plus;%5Calpha_%7Bj%7D%29%20%26%20%5Ctext%7B%20if%20%7D%20y%5E%7Bi%7D%3Dy%5E%7Bj%7D%20%5Cend%7Bcases%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;L and H are such that:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%5Cleq%20%5Calpha_%7Bi%7D%5Cleq%20H)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Calpha_%7Bj%7D%3A%3D%20%5Calpha_%7Bj%7D-%5Cfrac%7By_%7Bi%7D%28E_%7Bj%7D-E_%7Bi%7D%29%7D%7B%5Ceta%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where,   ![](http://latex.codecogs.com/gif.latex?%5Ceta%3D2K%28x_%7Bi%7D%2Cx_%7Bj%7D%29-K%28x_%7Bi%7D%2Cx_%7Bi%7D%29-K%28x_%7Bj%7D%2Cx_%7Bj%7D%29)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Cliping ![](http://latex.codecogs.com/gif.latex?%5Calpha_%7B1%7D) to lie within the range [-1,1].  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Calpha_%7Bj%7D%3A%3D%5Cbegin%7Bcases%7D%20H%26%20%5Ctext%7B%20if%20%7D%20%5Calpha_%7Bj%7D%3EH%20%5C%5C%20%5Calpha_%7Bj%7D%26%20%5Ctext%7B%20if%20%7D%20L%5Cleq%20%5Calpha_%7Bj%7D%5Cleq%20H%20%5C%5C%20L%26%20%5Ctext%7B%20if%20%7D%20%5Calpha_%7Bj%7D%3CL%20%5Cend%7Bcases%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Similarly for ![](http://latex.codecogs.com/gif.latex?%5Calpha_%7Bi%7D) and ![](http://latex.codecogs.com/gif.latex?B), we will update it by
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Calpha_%7Bi%7D%3A%3D%20%5Calpha_%7Bi%7D&plus;y_%7Bi%7Dy_%7Bj%7D%28%5Calpha_%7Bj%7D%5E%7Bold%7D-%5Calpha_%7Bi%7D%5E%7Bnew%7D%29)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?B%3A%3D%5Cbegin%7Bcases%7D%20B-E_%7Bi%7D-y_%7Bi%7D%28%5Calpha_%7Bi%7D%5E%7Bnew%7D-%5Calpha_%7Bi%7D%5E%7Bold%7DK%29-y_%7Bj%7D%28%5Calpha_%7Bj%7D%5E%7Bnew%7D-%5Calpha_%7Bj%7D%5E%7Bold%7D%29K%26%20%5Ctext%7B%20%2C%20if%20%7D%200%3C%5Calpha_%7Bi%7D%3CC%20%5C%5C%20B-E_%7Bj%7D-y_%7Bi%7D%28%5Calpha_%7Bi%7D%5E%7Bnew%7D-%5Calpha_%7Bi%7D%5E%7Bold%7DK%29-y_%7Bj%7D%28%5Calpha_%7Bj%7D%5E%7Bnew%7D-%5Calpha_%7Bj%7D%5E%7Bold%7D%29K%26%20%5Ctext%7B%20%2C%20if%20%7D%200%3C%5Calpha_%7Bj%7D%3CC%20%5C%5C%20%5Ctext%7Bavg.%20of%20two%20cases%20above%7D%20%26%20%5Ctext%7B%2C%20otherwise%7D%20%5Cend%7Bcases%7D)  

**Step 3**: Predicting the testing value. Classification labels are given by the armax of *y* along each column.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?ytest%3D%5Csum%20%5Calpha%5Cleft%20%28%20yK_%7Btest%7D%28x_%7Bi%7Dx_%7Bj%7D%29%20%5Cright%20%29&plus;B)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?y%3D%5Cbegin%7Bcases%7D%201%26%20%5Ctext%7B%20if%20%7D%20ytest%5Cgeq%200%5C%5C%20-1%26%20%5Ctext%7B%20if%20%7D%20ytest%3C0%20%5Cend%7Bcases%7D)  

**RETURN**: 
>*y_lbl*: Classification labels ('y') for the given test input.  
*y_tmp2*: Index matrix of respective test vaalue in label.
