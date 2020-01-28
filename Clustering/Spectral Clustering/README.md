# SPECTRAL CLUSTERING

**SIGNATURE**:   
>_spectral(X,K,**kwargs)_  

**DOCSTRING**:  
Perform spectral clutering with the given input training data and predict (y) label for clustering.   

**INPUT**:  
>*x*= Given the training data matrix in ndarray  
*k*= Number of cluster centers.  
_**kwargs_: 1) _laplacian_= '**symmetric**' For symmetric laplacian matrix, 2) _laplacian_='**random-walk**' For random-walk laplacian matrix 3) _laplacian_='**unnormalised**' For unnormalised laplacian matrix. Also, '**sigma** is the variance of the similarity matrix.  

**FUNCTION INSIDE THE CODE**:  
1) _kmean_pp(x,k)_: Peforms K-Mean with K-means++ for center initialisation.   
RETURN: *c* with same dimension like of *x*.  
2) _silhouetto(X,Y,K)_: To check the efficacy of the produced result using silhouetto score.  
RETURN: *s_avg*,The Silhouette score as an average of all the silhouette values.  
3) _similar(xi,xj)_: Calculate the similarity matrix of the given vectors as follows,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S%3De%5E%7B-%5Cfrac%7B%28x_%7Bi%7D-x_%7Bj%7D%29%5E%7B2%7D%7D%7B2%5Csigma%5E%7B2%7D%7D%20%7D)  
RETURN: S  

**PROCESS**:  
**Step 1**: Calculates the similarity matrix of every point with one another.  
**Step 2**: Form a degree matrix which is a diagonal matrix with diagonal elements as the sum of all the similarity values for a point.    
**Step 3**: Forming the Laplacian Matrix. The symmetric normalized Laplacian matrix is defined as,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%5E%7B%5Ctext%7Bsym%7D%7D%3A%3DI-B%5E%7B-1%7DSB%5E%7B-1%7D%2C%5Ctext%7Bwhere%2C%20B%20is%20the%20Cholesky%20Decomposition%20of%20D%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For Random-walk:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%5E%7Brwalk%7D%3A%3D%20I-%20D%5E%7B-1%7DS)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For unnormalised:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%3DD-S)    
**Step 4**: Calculate eigenvalues of L (Following steps reduce dimension, similar points comes closer with this step)  
**Step 5**: First k eignvalues(smallest).  
**Step 6**: Kmean clsutering  

**RETURN**:   
>*Y*: Cluster label.  
