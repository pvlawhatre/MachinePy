# GAUSSIAN PROCESSES REGRESSOR

**SIGNATURE**: 
>GP_regressor(x_train,y_train,x_test,sig_y,sig_f,l)

**DOCSTRING**:
Perform Gaussian processes regression with the given input training data and predict ( x_test data matrix) result
**INPUT**:
>x_train= Given the training data matrix in ndarray
y_train= Given the training label array in ndarray
x_test= Given the test data matrix in ndarray
sig_y= noise term in the diagonal of Ky. Should be zero if the data is noise free and greater than zero if not.
sig_f= Vertical variation.
l=length parameter to controls the smoothness of the function.

**FUNCTIONS INSIDE THE CODE**:
1) _Kernel(x,xp)_:	Kernel function for the GP as follows
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3D%5Csigma_%7Bf%7D%5E%7B2%7Dexp%5Cleft%20%28-%5Cfrac%7B%28x_%7Bi%7D-x_%7Bj%7D%29%5E%7BT%7D%28x_%7Bi%7D-x_%7Bj%7D%29%7D%7B2l%5E%7B2%7D%7D%20%5Cright%20%29)
RETURN: The kernel for the given input parameters.

**PROCESS**:
**Step 1**:Calculating Ky of shape (n_train×n_train) as follows:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K_%7By%7D%3DK%5Cleft%20%28x_%7Btrain%7D%5E%7Brepeat%7D%2Cx_%7Btrain%7D%5E%7Btile%7D%20%5Cright%20%29&plus;%5Csigma_%7By%7D%5E%7B2%7DI)
**Step 2**: Calculating K* of shape (n_train×n_test)as follows:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K_%7B*%7D%3DK%5Cleft%20%28x_%7Btrain%7D%5E%7Brepeat%7D%2Cx_%7Btrain%7D%5E%7Btile%7D%20%5Cright%20%29)
**Step 3**: Calculating K** of shape (n_test×n_test) as follows:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K_%7B**%7D%3DK%5Cleft%20%28x_%7Btrain%7D%5E%7Brepeat%7D%2Cx_%7Btrain%7D%5E%7Btile%7D%20%5Cright%20%29)
**Step 4**: Calculating mean and co-variance matrices:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cmu_%7B*%7D%3DK_%7B*%7D%5E%7BT%7DK_%7By%7D%5E%7B-1%7Dy)

**RETURN**: 
>mu: Mean of the Gaussian distribution
S:Variance of the Gaussian Distribution.