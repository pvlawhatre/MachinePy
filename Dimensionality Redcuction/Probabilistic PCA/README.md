# PROBABILISTIC PCA

**SIGNATURE**:   
>_prob_pca(X,K,sigma)_  

**DOCSTRING**:  
Performs probabilistic PCA over the given data matrix and return the gaussian distribuion parameters.   

**INPUT**:  
>*x*= Given the training data matrix in ndarray  
*k*= Desired dimension for the given data matrix.  
*Sigma*: σ parameters for Cholesky decomposition. 

**PROCESS**:  
**Step 1**: Mean and co-variance matrix calculation.  
**Step 2**: EVD of covariance matrix, Eigen values *val* and Eigen vectors *vec*.      
**Step 3**: Maximum likelihood estimation,    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?A%3DD_%7Bk%5Ctimes%20k%7D-%5Csigma%5E%7B2%7DI_%7Bk%5Ctimes%20k%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?A%3DLL%5E%7B*%7D%20%5Ctext%7B%2C%20%28Cholesky%20Decomposition%29%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?W%3DULI%20%5Ctext%7B%2C%20where%20U%3D%20Eigen%20Vector%20of%20top%20K%20Eigen%20values%7D)  
**Step 4**: Predicted mean of the distribution is as given as,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cmu_%7Bnew%7D%3D%28%28x-%5Cmu_%7Bx%7D%29%5Cbullet%20W%29%5Cbullet%20D)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Csigma%3D%5Cfrac%7BL%7D%7B%5Csigma_%7Binput%7D%5E%7B2%7D%7D)
**RETURN**:   
>*mean*:  Mean of the distribution from which the input point has come in the K reduced dimension.  
*cov*: Covariance of the distribution from which the input point has come in the K reduced dimension.  