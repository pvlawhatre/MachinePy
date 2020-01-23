# ISOMAP

**SIGNATURE**: 
>isomap(X,n_k,k)

**DOCSTRING**:
ISOMAP-Isometric Feature Maping, for dimensionality reduction also preserving the geodesic distances.
**INPUT**:
>X: Data matrix of dimension (n×d)
n_k: K value for KNN
k: Desired dimension for the given data matrix

**FUNCTIONS USED IN THE CODE**:
1) _nonisolated_G(n_k)_: To create a graph network of geodesic distances of the manifold.
      * _centering_mat(num)_: Input is an int data type number.
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?a2%3D%5Cbegin%7Bbmatrix%7D%201%26%200%26%200%26%20%5Ccdots%26%200%26%20%5C%5C%200%26%201%26%200%26%20%5Ccdots%26%200%26%20%5C%5C%200%26%200%26%201%26%20%5Ccdots%26%200%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Cvdots%26%20%5Cddots%26%20%5Cvdots%26%20%5C%5C%200%26%200%26%200%26%20%5Ccdots%26%201%26%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20n%29%7D)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?a2%3D%5Cfrac%7B1%7D%7Bn%7D%5Cbegin%7Bbmatrix%7D%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Cvdots%26%20%5Cddots%26%20%5Cvdots%26%20%5C%5C%201%26%201%26%201%26%20%5Ccdots%26%201%26%20%5C%5C%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20n%29%7D)
    RETURN:	 _H=a1-a2_
    * _geodesic(Net,source)_: Performing Dijkstra Algorithm to calculate the geodesic distances. (Read the [Dijsktra Algortihm][1] for more).
    RETURN: _dist_, geodesic distance of all points of shape (n×k).
    * _L2norm()_: Calculate the L2 norm for each and every points in the ‘d’ dimensions.
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L2%3D%5Cbegin%7Bbmatrix%7D%200%26%20&plus;%26%20&plus;%26%20%5Ccdots%26%20&plus;%26%20%5C%5C%20&plus;%26%200%26%20&plus;%26%20%5Ccdots%26%20&plus;%26%20%5C%5C%20&plus;%26%20&plus;%26%200%26%20%5Ccdots%26%20&plus;%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Cvdots%26%20%5Cddots%26%20%5Cvdots%26%20%5C%5C%20&plus;%26%20&plus;%26%20&plus;%26%20%5Ccdots%26%200%26%20%5Cend%7Bbmatrix%7D_%7B%28n%20%5Ctimes%20n%29%7D%2Cwhere%2C%27&plus;%27%20indicate%20the%20positive%20distance)
    RETURN: _L2_- A matrix of size (n×n) of distances of every point w.r.t. each other.
    * _graph(L2)_: Convert the distance matrix into a connected graph for K-neighbours.
     RETURN: _G_, a graph matrix of size (n×n) with values are only if the point doesn’t lie in the _K_ nearest neighbours.
     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RETURN: A semi-definate positive Gram matrix G

**PROCESS**:
**_Step 1_**: Geodesic distance calculation using Dijkstra algorithm. Returns ‘D’.
**_Step 2_**: Centering (no. of points), H and G is the Gram matrix which are:-
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?H%3DI-%5Cfrac%7B1%7D%7Bn%7D%5B%5Ctextbf%7B1%7D%5D_%7B%28n%5Ctimes%20n%29%7D) ,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?tmp%3D-%5Cfrac%7B1%7D%7B2%7DD%5E%7B2%7D%28%5Ctextup%7BGower%20Transformation%7D%29)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?G%3D%28H%5Ccdot%20tmp%29%5Ccdot%20H)
**_Step 3_**:  EVD decomposition- Try and except block will make the n_k sufficient enough for the Eigen values and the corresponding Eigen vectors calculation.
**_Step 4_**: Projection matrix- Eigen values and the Eigen vectors are sorted and the top k value and the corresponding vector are taken into account for the final reduced Y.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Y%3D%5Clambda%20%5E%7B1/2%7DV) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where, λ= val, V=pvec,val=Eigen value of top k values of G,pvec=Eigen vectors of corresponding values.

**RETURN**:
> Returns Y, a reduced matrix of dimensions (n×k).

[1]:https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm