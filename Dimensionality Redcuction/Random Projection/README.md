# RANDOM PROJECTION

**SIGNATURE**: 
>random_projection(x,k,*args)

**DOCSTRING**:
Performs random projection over the given data matrix.
**INPUT**:
>x=Data matrix of dimension (n×d)
K=Desired dimension for the given data matrix
*arg: Arg have two types- for Gaussian Random Matrix and Sparse Random 	Matrix. It must be given as list or a string as ‘**gauss**’ and 	‘**sparse**’ respectively.

**PROCESS**:
1) *args=’ **_gauss_** ’: **input function**--> **_gaussian_ranodm_matrix(m,n)_**
**Input**: m=old dimension, n= new dimension
RETURN: p, a gaussian random matrix of dimensions (d×k)
2) *arg=’ **_sparse_**’:	**input function**--> **_sparse_random_matrix(m,n)_**
**Input**: m=old dimension, n= new dimension
RETURN: p, a sparse random matrix of dimensions (d×k)

Finally, the _y_ nwill be calculated as:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?y%3Dxt%5Cbullet%20p)
**RAISE**: 
 For training other than ‘gauss’ and ‘sparse’, error would be generated.
**RETURN**: 
> y, a reduced matrix of dimensions (n×k)
    
