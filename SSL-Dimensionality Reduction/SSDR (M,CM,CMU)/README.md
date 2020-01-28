# SSDR-M,SSDR-CM, SSDR-CMU

**SIGNATURE**:   
>_SSDR_constrained(X,K,**kwargs)_    

**DOCSTRING**:  
Perform SSDR with must link, cannot link and using both with the given input training data and reduced into desired dimension

**INPUT**:  
>*X*= Given the training data matrix in ndarray  
*K*= Desired dimension to be acheived.  
_**kwargs_= 1) '_**method**_'='**M**', for using must link only, 2) '**_method_**'='**CM**', for using cannot link and must link, 3) '**_beta_**'=10, (DEFAULT) for must link method, 4) '**_alpha_**', for CMU mehtod.   
*ml*= Data points that are to be together.  
*cl*= Data points that cannot be together.  

**PROCESS**:  
**Step 1**: Determining points in must-link and cannot-link  
**Step 2**: Calculating the weight matrix, *S* as  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S_%7Bij%7D%3D%5Cbegin%7Bcases%7D%20%5Cfrac%7B-%5Cbeta%7D%7Bn_%7BM%7D%7D%26%20%5Ctext%7B%20if%20%7D%20%28x_%7Bi%7D%2Cx_%7Bj%7D%29%5Cepsilon%20M%20%5C%5C%200%26%20%5Ctext%7B%20otherwise%20%7D%20%5Cend%7Bcases%7D%5Ctext%7B%2C%28For%20must%20link%20method%20only%29%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S_%7Bij%7D%3D%5Cbegin%7Bcases%7D%20%5Cfrac%7B%5Calpha%7D%7Bn_%7BC%7D%7D%26%20%5Ctext%7B%20if%20%7D%20%28x_%7Bi%7D%2Cx_%7Bj%7D%29%5Cepsilon%20C%20%5C%5C%20%5Cfrac%7B-%5Cbeta%7D%7Bn_%7BM%7D%7D%26%20%5Ctext%7B%20if%20%7D%20%28x_%7Bi%7D%2Cx_%7Bj%7D%29%5Cepsilon%20M%20%5C%5C%200%26%20%5Ctext%7B%20otherwise%20%7D%20%5Cend%7Bcases%7D%5Ctext%7B%2C%28For%20must%20link%20and%20cannot%20link%20method%29%7D)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S_%7Bij%7D%3D%5Cbegin%7Bcases%7D%20%5Cfrac%7B1%7D%7Bn%5E%7B2%7D%7D&plus;%5Cfrac%7B%5Calpha%7D%7Bn_%7BC%7D%7D%26%20%5Ctext%7B%20if%20%7D%20%28x_%7Bi%7D%2Cx_%7Bj%7D%29%5Cepsilon%20C%20%5C%5C%20%5Cfrac%7B1%7D%7Bn%5E%7B2%7D%7D-%5Cfrac%7B%5Cbeta%7D%7Bn_%7BM%7D%7D%26%20%5Ctext%7B%20if%20%7D%20%28x_%7Bi%7D%2Cx_%7Bj%7D%29%5Cepsilon%20M%20%5C%5C%20%5Cfrac%7B1%7D%7Bn%5E%7B2%7D%7D%26%20%5Ctext%7B%20otherwise%20%7D%20%5Cend%7Bcases%7D%5Ctext%7B%2C%28For%20must%20link%20and%20cannot%20link%20method%20on%20unlabelled%20data%29%7D)  
**Step 3**: Calculating the Laplacian matrix and EVD for ![](http://latex.codecogs.com/gif.latex?A%3DX%5E%7BT%7DLX). Top K eigen vectors are taken as projection matrix.  
**Step 4**: Transformation by taking the dot product of X and the projection matrix.   

**RETURN**:   
>*Y*: Test prediction unlabelled data.  
