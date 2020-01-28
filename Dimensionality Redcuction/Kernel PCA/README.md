# KERNEL- PCA

**SIGNATURE**: 
>_kernel_PCA(X,k,*args,**kwargs_

**DOCSTRING**:
Performs kernel PCA over the given data matrix.

**INPUT**:  
>*X*= Data matrix of dimension (n×d)  
*k*= Desired dimension for the given data matrix  
_*arg_: Arg have five input types- COSINE, LINEAR, POLY, SIGMOID 	and RBF for cosine function, linear function, polynomial function, 	sigmoid function and radial basis function respectively. It must be given 	as list or a string. **‘cosine’**, **’linear’** , **’poly’**, **’sigmiod’** and **‘rbf’** respectively.  
_**kwargs_: Kwarg values for the respective arg functions are below:-  

Funtions | Input 
--------- | ------------
‘cosine’ | NOTHING. LEAVE IT EMPTY
‘linear’ | NOTHING LEAVE IT EMPTY
‘poly’ | gamma=[int], c0=[int],degree=[int]
‘sigmoid’ | gamma=[int], c0=[int]
‘rbf’ | gamma=[int]

**FUNCTIONS USED IN THE CODE**:  
1) _cosine_similarity(x,y)_: When *arg is ‘cosine’. 
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3D%5Cfrac%7Bx_%7B1%7D%5E%7BT%7Dx_%7B2%7D%7D%7B%28%5Csqrt%7B%5Csum_%7Bi%3D_%7B1%7D%5E%7B1%7D%5Ctextrm%7Bx%7D%7D%5E%7B_%7B1%7D%5E%7Bn%7D%5Ctextrm%7Bx%7D%7Dx_%7Bi%7D%5E%7B2%7D%7D%29%28%5Csqrt%7B%5Csum_%7Bj%3D_%7B2%7D%5E%7B1%7D%5Ctextrm%7Bx%7D%7D%5E%7B_%7B2%7D%5E%7Bn%7D%5Ctextrm%7Bx%7D%7Dx_%7Bj%7D%5E%7B2%7D%7D%29%7D)  
    RETURN: The kernel _K_ of dimension same as of x.  
    
2) _linear(x,y)_: When *arg is ‘linear’.  
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3Dx_%7B1%7D%5E%7BT%7Dx_%7B2%7D)  
RETURN: The kernel _K_ of dimension same as of x.  
3) _poly(x,y)_:	When *arg is ‘poly’.  
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3D%28%28x_%7B1%7D%5E%7BT%7Dx_%7B2%7D%29%5Cgamma&plus;c_%7B0%7D%29%5E%7Bdeg%7D)  
RETURN: The kernel _K_ of dimension same as of x.  
4) _sigmoid(x,y)_:	When *arg is ‘sigmoid’.  
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3D%5Ctanh%20%28%28x_%7B1%7D%5E%7BT%7Dx_%7B2%7D%29%5Cgamma&plus;c_%7B0%7D%29)  
RETURN: The kernel _K_ of dimension same as of x.  
5) _rbf(x,y)_:	When *arg is ‘rbf’.  
Input:	x: A point in the data matrix with ‘d’ dimensions.  
y: A point in the data matrix with ‘d’ dimensions.  
Governing equation is->  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3Dexp%28-%5Cleft%20%7C%20x_%7B1%7D%5E%7BT%7D-x_%7B2%7D%20%5Cright%20%7C%5E%7B2%7D%5Cgamma%29)  
RETURN: The kernel _K_ of dimension same as of x.  
6) coding_mat(y):	Take 1D array of label of X as input. Following function takes place-  
For every point in class-  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Z%5Bi%2Cj%5D%3D%28%5Csqrt%7B%5Cfrac%7Bn%7D%7BC%7D%7D%29-%28%5Csqrt%7B%5Cfrac%7BC%7D%7Bn%7D%7D%29)  
    Else-  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Z%5Bi%2Cj%5D%3D-%28%5Csqrt%7B%5Cfrac%7BC%7D%7Bn%7D%7D%29)  
    Where, n=No. of points, C=No. of points in a class.  
RETURN: The coding matrix with dimensions as (n×no. of classes)  

**PROCESS**:  
**_Step 1_**: 	Kernel formation  
**_Step 2_**:  Centralised Kernel:-  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3DK-%28%5Cfrac%7B1%7D%7Bn%7DI%5Cbullet%20K%29-%28K%5Cbullet%20%5Cfrac%7B1%7D%7Bn%7DI%29&plus;%28%28%5Cfrac%7B1%7D%7B2%7DI%5Cbullet%20K%29%5Cbullet%20%5Cfrac%7B1%7D%7Bn%7DI%29)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where, I has the dimension same as of K  
**_Step 3_**:  Eigen Value Decomposition of Kernel  
**_Step 4_**: Rescaling eignvectors  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?vec%3D%5Cfrac%7B1%7D%7B%5Csqrt%7Bval%5Ctimes%20n%7D%7Dvec)  
**_Step 5_**:  Projection  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?y%3DK%5Cbullet%20vec)  

**RETURN**:  
>*Y*, a reduced matrix of dimensions (n×k). 
