# LINEAR DISCRIMINANT ANALYSIS (LDA)

**SIGNATURE**: 
>LDA(x,y,k)

**DOCSTRING**:
Performs LDA to a given data matrix x.
**INPUT**:
>x: Data matrix of dimension (n×d)
y: Label array of X (1D)
k: Desired dimension for the given data matrix

**PROCESS**:
**_Step 1_**: 	Mean calculation:-
1) Overall mean, **_m_** 
2) Class mean, **_mc_**.

**_Step 2_**:  Scatter matrix:-
1) Within class,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S_%7Bj%7D%3D%5Csum%20%28x_%7Bk%7D-%5Cbar%7Bx_%7Bj%7D%7D%29%28x_%7Bk%7D-%5Cbar%7Bx_%7Bj%7D%7D%29%5E%7BT%7D%2C%5C%3B%20x_%7Bk%7D%5Cepsilon%20X_%7Bj%7D)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S_%7Bw%7D%3D%5Csum_%7Bj%3D1%7D%5E%7Bc%7DS_%7Bj%7D)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where, Xj is the set of points in a particular cluster and C is the number of clusters.
2) Between- class scatter matrix
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S_%7BB%7D%3D%5Csum_%7Bj%3D1%7D%5E%7Bc%7D%5Cleft%20%7C%20X_%7Bj%7D%20%5Cright%20%7C%28%5Cbar%7Bx_%7Bj%7D%7D-mc%29%5Cbullet%28%5Cbar%7Bx_%7Bj%7D%7D-mc%29%5E%7BT%7D)

**_Step 3_**:  Eigen Value Decomposition-
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S%3DS_%7Bw%7D%5E%7B-1%7D%5Cbullet%20S_%7BB%7D), then the classic Eigen value problem will become as following:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%28S-%5Clambda%29W%3D0)
**_Step 4_**: Projection and transformation.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?p%3Dvec%5B%3A%2C0%3Ak%5D%2C%5C%3B%20top%5C%3B%20k%5C%3B%20Eigen%5C%3B%20Values%5C%3B%20are%20%5C%3B%20taken)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?xtrans%3Dx%5Cbullet%20p)

**RETURN**:
>X_trans, a reduced matrix of dimensions (n×k). 