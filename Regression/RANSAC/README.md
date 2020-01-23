# RANDOM SAMPLE CONSENSUS (RANSAC)

**SIGNATURE**: 
>ransac(x_train,y_train,x_test,min_pts,sigma,min_iter)

**DOCSTRING**:
Perform RANSAC with the given input training data and predict ( x_test data matrix) result.
**INPUT**:
>x_train= Given the training data matrix in ndarray
y_train= Given the training label array in ndarray
x_test= Given the test data matrix in ndarray
min_pts= Minimum number of points for regression
sigma= Variance for the given input data
min_iter= Minimum number of iterations.

**PROCESS**:
**Step 1**: Randomly select minimum points for regression.
**Step 2**: Perform regression and finding coefficients.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Xin%3D%5Cbegin%7Bbmatrix%7D%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20xtrain%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cend%7Bbmatrix%7D_%7B%28min.pts%5Ctimes%20d&plus;1%29%7D)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Here, x-train is of randomly selected point from x_train.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Cin%3DXin%5E%7B-1%7D%5Cbullet%20ytrain)&nbsp;&nbsp;also, *y_train* is of randomly selected point from *y_train*.
**Step 3**: Inliers:-
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Xtrain%3D%5Cbegin%7Bbmatrix%7D%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20xtrain%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20d&plus;1%29%7D)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Din%3D%5Cleft%20%7C%20%5Cfrac%7BXtrain%5Cbullet%20Cin-ytrain%7D%7B%5Csqrt%20%281&plus;%5Cleft%20%7C%20Cin%20%5Cright%20%7C%5E%7B2%7D%29%7D%20%5Cright%20%7C)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This, is the error term. Thresholding by the *Din≤Sigma*, the inliers for the randomly selected points are saved &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in inlier of shape (n_train×1).
**Step 4**:Model selection. Maximum number of inlier is selected. xin and yin are the x_train and y_train inlier values &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;respectively.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Xin%3D%5Cbegin%7Bbmatrix%7D%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20xtrain%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20d&plus;1%29%7D)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Cin%3DXin%5E%7B-1%7D%5Cbullet%20yin)
**Step 5**: Predicting the y_test of the given x_test and returning it.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?Xt%3D%5Cbegin%7Bbmatrix%7D%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Cvdots%26%20%5Ccdots%26%20xtest%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%20%5Cvdots%26%20%5Cvdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cvdots%26%20%5C%5C%201%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Ccdots%26%20%5Cend%7Bbmatrix%7D_%7B%28n%5Ctimes%20d&plus;1%29%7D)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?ytest%3DXt%5Cbullet%20Cin)

**RETURN**: 
>xin= Number of inliers in the fitted model for x_train
yin= Number of inliers in the fitted model for y_train. 
Cin: Coefficient matrix with shape (d+1×1)
y_test: test prediction with shape similar to y_train
