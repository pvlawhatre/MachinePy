# GAUSSIAN MIXTURE MODEL

**SIGNATURE**:   
>_GMM(x,K)_  

**DOCSTRING**:  
Perform gaussian Mixture Model with the given input training data and predict (y) label for clustering.   

**INPUT**:  
>*x*= Given the training data matrix in ndarray  
*k*= Number of cluster centers.  

**FUNCTION INSIDE THE CODE**:  
1) _kpp(X,K)_: K-means++ for center initialisation.  
RETURN: *c* with same dimension like of *x*.  

2) *mvg(x,M,H)*: Calculates multi-variate Gaussin  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?y%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%5E%7Bd%7D%5Csqrt%7B%5Cleft%20%7C%5Csum%20%5Cright%20%7C%7D%7Dexp%5Cleft%20%28%20-%5Cfrac%7B%28x-%5Cmu%29%5E%7BT%7D%5Csum%5E%7B-1%7D%28x-%5Cmu%29%7D%7B2%7D%20%5Cright%20%29)  
RETURN: *y*    

**PROCESS**:  
**Step 1**: Initializing centroid with kmean++, covariance matrix to be identity matrix and prior 

**Step 2**: Applying Expectation step considering the points comes from a multi-variate gaussian distribution. 

**Step 3**: Applying Maximisation step.     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cmu_%7Bi%7D%3D%5Cfrac%7By_%7Bji%7Dx_%7Bij%7D%7D%7B%5Cmu_%7By%7D%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Csum_%7Bi%7D%3D%5Cleft%20%28%20x-%5Cmu_%7Bi%7D%20%5Cright%20%29%5E%7BT%7D%5Cleft%20%28%20%28x-%5Cmu_%7Bi%7D%29%28%5Cfrac%7By_%7Bk%7D%7D%7Bn%5Cmu_%7By%7D%7D%5Cmathbf%7B1%7D_%7Bk%5Ctimes%20n%7D%29%5E%7BT%7D%20%5Cright%20%29)

**Step 4**: Repeat until the convergence.   

**RETURN**:   
>*Y*: Expectation values.  
*lbl*:  Cluster label.  
*mu*: Centroid matrix of dimension (K,1).  
