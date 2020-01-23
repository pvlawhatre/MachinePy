# CLASSICAL MULTI(MATRIX)-DIMENSIONAL SCALING (cMDS)

**SIGNATURE**: 
>*cMDS(x,k)*

**DOCSTRING**:
Performs Multi-Dimensional Scaling over the given input data matrix.
**INPUT**:
>_x_: Data matrix of dimension (n×d)
_K_: Desired dimension for the given data matrix

**FUNCTIONS USED IN THE CODE**:
1) _centring_mat(num)_: Input is an int data type number. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?a2%3D%5Cbegin%7Bbmatrix%7D%201%26%200%26%200%26%20%5Ccdots%26%200%26%20%5C%5C%200%26%201%26%200%26%20%5Ccdots%26%200%26%20%5C%5C%200%26%200%26%201%26%20%5Ccdots%26%200%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Cvdots%26%20%5Cddots%26%20%5Cvdots%26%20%5C%5C%200%26%200%26%200%26%20%5Ccdots%26%201%26%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20n%29%7D)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?a2%3D%5Cfrac%7B1%7D%7Bn%7D%5Cbegin%7Bbmatrix%7D%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Cvdots%26%20%5Cddots%26%20%5Cvdots%26%20%5C%5C%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20n%29%7D)
RETURN:	 H=a1-a2

**PROCESS**:
**_Step 1_**: Proximity matrix/ Dissimilarity matrix calculation- D[i,j] is the distance of one point to with the other points in ‘d’ dimension. Here the distance is Euclidean distance only but the distance could be anything else.
**_Step 2_**: Centering (no. of points), H and G is the Gram matrix which are:-
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?H%3DI-%5Cfrac%7B1%7D%7Bn%7D%5B%5Ctextbf%7B1%7D%5D_%7B%28n%5Ctimes%20n%29%7D) ,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?tmp%3D-%5Cfrac%7B1%7D%7B2%7DD%5E%7B2%7D%28%5Ctextup%7BGower%20Transformation%7D%29)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?G%3D%28H%5Ccdot%20tmp%29%5Ccdot%20H)
**_Step 3_**: EVD decomposition- Eigen values (val) and the corresponding Eigen Vectors(vec) of G.
**_Step 4_**: Projection matrix- Eigen values and the Eigen vectors are sorted and the top k value and the corresponding vector are taken into account for the final reduced Y.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Y%3D%5Clambda%20%5E%7B1/2%7DV) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where, λ= val, V=pvec,val=Eigen value of top k values of G,pvec=Eigen vectors of corresponding values.

**RETURN**:
> Returns Y, a reduced matrix of dimensions (n×k).
