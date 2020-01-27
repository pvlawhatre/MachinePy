# MEAN-SHIFT

**SIGNATURE**:   
>_mean_shift(x)_

**DOCSTRING**:  
Perform mean-shift with the given input training data and predict (y) label for clustering.  

**INPUT**:  
>*x*= Given the training data matrix in ndarray  

**FUNCTION INSIDE THE CODE**:  
1) _gauss(x)_ : Gauss function as follows, 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?f%28x%29%3De%5E%7B-%5Cfrac%7B%5Cleft%20%7C%20x%20%5Cright%20%7C%5E%7B2%7D%7D%7B2%7D%7D)  
RETURN: A Gaussian function.   

2) _genC(x)_: Generates each points with added jitter (Gaussian noise with meann 0.5 and standard deviation 1).  
RETURN: *c* with same dimension like of *x*.  

3) *jitter(x)*: Add gaussian noise into the given point and increases jitterness.  
RETURN: *v* a jittered point.  

**PROCESS**:  
**Step 1**: Initializing random points  
**Step 2**: Calculating the center of gravity  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?C%3D%5Cfrac%7B%5Csum_%7Bi%7D%20%5Csum_%7Bj%7D%20N%28C_%7Bi%7D%2C1%29*x_%7Bj%7D%7D%7B%5Csum%20N%28C%2C1%29%7D)  
**Step 3**: Repeat until convergence.   
**Step 4**: Shift the search window to the mean.     

**RETURN**:   
>*y*: Predicted class label for the respective points.  
*C*: CLuster points.