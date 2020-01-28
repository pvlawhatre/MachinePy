# LAGUERRE FUNCTIONAL LINK NEURAL NETWORK (FLNN)

**SIGNATURE**:  
>_LFLNN(X_train,y_train,X_test,l,eta,**kwargs)_    

**DOCSTRING**:  
Perform LFLNN to the given input training data and predict the test data for regression.

**INPUT**:  
>*X_train*= Given the input training data matrix in ndarray  
*Y_train*= Given the output labelled data matrix in ndarray 
*X_test*= Given the input test data matrix in ndarray 
*l*= Order of Laguerre polynomial 
*eta*= Learning rate 
_**kwargs_= 1) '*eps*': Thershold value, 2) '*stable*': Flag count, 3) '*itr*': Numnber of iterations.  

**FUNCTION INSIDE THE CODE**:  
1) _laguerre(l,x)_: Legendre polynomial function as follows:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L_%7B0%7D%28x%29%3D1%2CL_%7B1%7D%28x%29%3D1-x)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L_%7Bn&plus;1%7D%28x%29%3D%5Cfrac%7B%282n&plus;1-x%29L_%7Bn%7D%28x%29-nL_%7Bn-1%7D%28x%29%7D%7Bn&plus;1%7D)  

**PROCESS**:  
**Step 1**: Function expansion block in which a new training and testing set is formed with dimension similar to the training and testing set but with extra number of columns which is equal to the order of the Laguerre Polynomial. Laguerre polynomial is calculated of the training and testing dataset and then appended in the new training and testing data sets.
**Step 2**: Training of the model.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?X%3DT_%7Btrain%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Chat%20Y%3DW%5E%7BT%7DX&plus;%5Cmu_%7BX%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?W_%7Bnew%7D%3DW_%7Bold%7D&plus;%5Ceta%28Y-%5Chat%20Y%29X)  
**Step 3**: Repeated till the threshold is reached or the given number of iterations is reached.  
**Step 4**: Prediction is don as follows,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Chat%20Y%3DW%5E%7BT%7DX_%7Btest%7D&plus;%5Cmu_%7BX_%7Btest%7D%7D)

**RETURN**:   
>*Y*: Test data set for the given data data inputs.  
*W*: Weight matrix of the regressor.  