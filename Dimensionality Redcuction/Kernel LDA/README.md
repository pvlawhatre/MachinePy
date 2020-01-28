# KERNEL- LDA

**SIGNATURE**: 
>_kernel_LDA(X,Y,k,*args,**kwargs)_

**DOCSTRING**:
Performs kernel LDA into the given data matrix.

**INPUT**:
>*X*= Data matrix of dimension (n×d)  
*Y*= Label array of X (1D)  
*k*= Desired dimension for the given data matrix  
_*arg_: Arg have five input types- COSINE, LINEAR, POLY, SIGMOID 	and RBF for cosine function, linear function, polynomial function, 	sigmoid function and radial basis function respectively. It must be given 	as list or a string. **‘cosine’**, **’linear’** , **’poly’**, **’sigmiod’** and **‘rbf’** respectively.
**kwargs: Kwarg values for the respective arg functions are below:-  

Funtions | Input 
--------- | ------------
‘cosine’ | NOTHING. LEAVE IT EMPTY
‘linear’ | NOTHING LEAVE IT EMPTY
‘poly’ | gamma=[float32], c0=[float32],degree=[int]
‘sigmoid’ | gamma=[float32], c0=[float32]
‘rbf’ | gamma=[float32]

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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?J%3DI-%5Cfrac%7B1%7D%7Bn%7D%5Cmathbf%7B1%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where I is the identity matrix and 1 is the ones matrix (n×n)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?K%3DJKJ)  
**_Step 3_**:  Total scatter matrix  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?St%3DKK)  
**_Step 4_**: Between class scatter matrix and coding matrix  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Z%3DCoding%5C%3B%20Matrix)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Sb%3D%5Cfrac%7B1%7D%7Bn%7D%28KZZ%5E%7BT%7DK%29)  
**_Step 5_**: Within class scatter matrix  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Sw%3DSt-Sb)  
**_Step 6_**: Ratio  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S%3DS_%7Bw%7D%5E%7B-1%7DS_%7Bb%7D)  
**_Step 7_**: EVD. Then the classic Eigen value problem will become as following:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%28S-%5Clambda%29W%3D0)  
**_Step 8_**: projection matrix  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?p%3Dvec%5B%3A%2C0%3Ak%5D%2Ctop%5C%3B%20k%5C%3B%20Eigen%5C%3B%20Value%5C%3B%20are%5C%3B%20taken)  
**_Step 9_**: Transformation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?xtrans%3DK%5Cbullet%20p)  

**RETURN**:
>*X_trans*, a reduced matrix of dimensions (n×k).
