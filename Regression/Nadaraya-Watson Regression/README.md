# NADARAYA-WATSON RERGRESSION

**SIGNATURE**: 
>_nadaraya_watson(X_train,Y_train,X_test,**kwargs)_

**DOCSTRING**:  
Perform Nadaraya-Watson regression with the given input training data and predict ( x_test data matrix) result. It is can have automatic or custom bandwidth.  

**INPUT**:  
>*X_train*= Given the training data matrix in ndarray  
*Y_train*= Given the training label array in ndarray  
*X_test*= Given the test data matrix in ndarray  
_**kwargs_: 1) Method=’silverman’:- Performs Silverman technique to calculate bandwidth  	
2)Method=’scott’:- Performs Scott technique to calculate bandwidth as follows   
3)BW=c:- c is an float type ndarray of shape (d×d)  

**FUNCTIONS INSIDE THE CODE**:  
1) *bw(strg,x)*: Calculate the bandwidth. Given the method type either Silverman or Scott, it calculate the ‘H’ as follows:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?H%3D%5Csigma%5E%7B2%7Dn%5E%7B%5Cfrac%7B-1%7D%7Bn&plus;4%7D%7D%5Cleft%20%28%5Cfrac%7B4%7D%7Bd&plus;2%7D%20%5Cright%20%29%5E%7B%5Cfrac%7B1%7D%7Bd&plus;4%7D%7D%2C%20%28Silverman%29)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?H%3D%5Csigma%5E%7B2%7Dn%5E%7B%5Cfrac%7B-1%7D%7Bn&plus;4%7D%7D%2C%5C%3B%20%28Scott%29)  
RETURN: H a ndarray of (d×d)  
2) *kernel(H,x)*: Kernel function with returning kernel as follows:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3D%5Cfrac%7B1%7D%7B%5Cleft%20%282%5Cpi%20%5Cright%20%29%5E%7Bd/2%7D%7D%5Cleft%20%7C%20H%20%5Cright%20%7C%5E%7B-1/2%7Dexp%5Cleft%20%28%20%5Cfrac%7B-1%7D%7B2%7Dx%5E%7BT%7DH%5E%7B-1%7Dx%20%5Cright%20%29)  
RETURN:  K of float type.  

**PROCESS**:  
**Step 1**: Calculate the bandwidth either by the above mentioned methods or a custom one in H.  
**Step 2**: Predict the values for the given test data as follows:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Y%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20K_%7Bh%7D%28x-x_%7Bi%7D%29y_%7Bi%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7Bn%7DK_%7Bh%7D%28x-x_%7Bj%7D%29%7D)  
The above equation is Nadaraya-Watson equation. Here, training sets is substracted from test set.  

**RETURN**:   
>*y_test*: test prediction with shape similar to y_train.  
