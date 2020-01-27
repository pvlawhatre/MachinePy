# DISCRIMINENT NEIGHBOURHOOD EMBEDDING

**SIGNATURE**:   
>_DNE(X,Y,K)_  

**DOCSTRING**:  
Performs DNE over the given data matrix for dimensionality reduction.  

**INPUT**:  
>*X*= Given the labelled input data matrix in ndarray  
*Y*= Given the labelled output data matrix in ndarray  
*k*= Desired K dimension 

**PROCESS**:  
**Step 1**:  Neighbourhood of the points
**Step 2**: Laplacian of the graph.      
**Step 3**: Eigen Value Dcomposition of following 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?C%3DX%5E%7BT%7DLX)  
**Step 4**: Transformation of projection matrix.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Y%3DXP)   

**RETURN**:   
>*y*: Reduced data points to the given K.