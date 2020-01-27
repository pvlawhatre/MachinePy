# SSDR-Manifold

**SIGNATURE**:  
>_SSDR_manifold(Xl,Yl,Xu,K,gamma)_    

**DOCSTRING**:  
Perform SSDR manifold to the given input training data and reduced into desired dimension

**INPUT**:  
>*Xl*= Given the input labelled data matrix in ndarray  
*Yl*= Given the output labelled data matrix in ndarray 
*Xu*= Given the input unlabelled data matrix in ndarray 
*K*= Desired dimension to be acheived.  
*gamma*= Scaling factor for the Gaussian  

**FUNCTION INSIDE THE CODE**:  
1) _gauss(xi,xj,si,sj)_: Return the gaussain with *si* and *sj* variance.  

**PROCESS**:  
**Step 1**: Labelled cost function which is the distance between the points to one another.  
**Step 2**: Unlabelled cost function.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?C%3DC_%7Bl%7D&plus;%5Cgamma%20exp%5Cleft%20%28%20-%5Cfrac%7B%28x_%7Bi%7D-x_%7Bj%7D%29%5E%7B2%7D%7D%7B%5Csigma_%7Bi%7D%20%5Csigma_%7Bj%7D%7D%20%5Cright%20%29)
**Step 3**: Calculating the Laplacian matrix and EVD for ![](http://latex.codecogs.com/gif.latex?T%3DX%5E%7BT%7DLX). Top K eigen vectors are taken as projection matrix.  
**Step 4**: Transformation by taking the dot product of X and the projection matrix.   

**RETURN**:   
>*Y*: Data matrix reduced to the desired dimension.  