# PRINCIPAL COMPONENT ANALYSIS (PCA)

**SIGNATURE**: 
>_pca(X,K,*arg)_

**DOCSTRING**:  
Performs PCA into the given data matrix X.

**INPUT**:  
>*X*: Data matrix of dimension (n×d)  
*K*:Desired dimension for the given data matrix  
_*arg_: Arg have two types- **_EVD_** for Eigen Value Decomposition and **_SVD_** for Singular Value Decomposition. It must be given as list or a string as ‘evd’ and ‘svd’ respectively.  

**PROCESS**:  
1) *args=’ **_evd_** ’:	**input function**--> **_EVD_pca(x,k)_**  
**_Step 1_**: Mean calculation and normalisation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?xt%3Dx-%5Cmu%2C%5C%3B%20where%5C%3B%20%5Cmu%5C%3B%20is%5C%3B%20mean)  
**_Step 2_**: Co-variance matrix  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?c%3D%5Cfrac%7B1%7D%7Bn-1%7D%28xt%5E%7BT%7D%5Cbullet%20xt%29%2C%5C%3B%20where%5C%3B%20n%3DNumber%5C%3B%20of%5C%3B%20data%5C%3B%20points%5C%3B%20in%5C%3B%20X)  
**_Step 3_**: Eigen Value Decomposition of _c_.  
**_Step 4_**: Projection matrix, projects the top ‘k’ eigen values and their corresponding Eigen vectors, _p_.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?y%3Dxt%5Cbullet%20p)  
2) *arg=’ **_svd_** ’:	**input function**--> **_SVD_pca(x,k)_**  
**_Step 1_**:  Mean calculation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?xt%3Dx-%5Cmu%2C%5C%3B%20where%5C%3B%20%5Cmu%5C%3B%20is%5C%3B%20mean)  
**_Step 2_**:  Singular Value Decomposition of x_t.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S%3DS_%7Bw%7D%5E%7B-1%7D%5Cbullet%20S_%7BB%7D)  

**RAISE**:  
If nothing is given, an error will occur and will show ‘_unidentified method_'.  

**RETURN**:  
>*y*, a reduced matrix of dimensions (n×k)
