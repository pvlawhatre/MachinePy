# FUZZY C-MEAN

**SIGNATURE**:   
>_cmeans(x,k,m)_  

**DOCSTRING**:  
Perform fuzzy c-mean with the given input training data and predict (y) label for clustering.   

**INPUT**:  
>*x*= Given the training data matrix in ndarray  
*k*= Number of cluster centers.
*m*= Fuzziness parameter.  

**FUNCTION INSIDE THE CODE**:  
1) _genC(x)_: Generates each points with added jitter (Gaussian noise with meann 0.5 and standard deviation 1).  
RETURN: *c* with same dimension like of *x*.  

2) *jitter(x)*: Add gaussian noise into the given point and increases jitterness.  
RETURN: *v* a jittered point.  

**PROCESS**:  
**Step 1**: Initializing random points and membership values.
**Step 2**: Update Centroid and weights 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%5Cgamma%3D%5Cfrac%7B1%7D%7B%5Csum_%7B1%7D%5E%7Bn%7D%5Cleft%20%28%5Cfrac%7Bx_%7Bi%7D-c_%7Bk%7D%7D%7Bx_%7Bj%7D-c_%7Bj%7D%7D%20%5Cright%20%29%5E%7B2/m-1%7D%7D)  
**Step 3**: Centroid update.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?C_%7Bij%7D%3D%5Cleft%20%28%5Cfrac%7B%5Csum_%7B1%7D%5E%7Bn%7D%20W_%7Bik%7D%5E%7Bm%7D%5Cbullet%20x_%7Bk%7D%7D%7B%5Csum_%7B1%7D%5E%7Bn%7D%20W_%7Bik%7D%5E%7Bm%7D%7D%20%5Cright%20%29)  
**Step 4**: Repeat until the convergence of the membership values.   

**RETURN**:   
>*Y*: Predicted class label for the respective points.  
*W*: Membership matrix of dimension (n,k)  
*C*: Centroid matrix of dimension (k,1)