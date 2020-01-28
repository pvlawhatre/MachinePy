# K-MEAN (++)

**SIGNATURE**:   
>_kmean_pp(x, K)_

**DOCSTRING**:  
Perform K-mean with the given input training data with K-mean++ algorithm and predict (y) label for clustering.  

**INPUT**:  
>*x*= Given the training data matrix in ndarray  
*k*= Mimimum K distance value.

**FUNCTION INSIDE THE CODE**:  
1) _kpp(x,K)_ : To initialise the center points. Following is the algorithm for the kmean++:-  
        1) Select 1st point randomly from a cluster.  
        2) For each data point compute its distance from the nearest, previously choosen centroid.  
        3) Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most likely to be selected next as a centroid).  
        4) Repeat steps 2 and 3 untill k centroids have been sampled.  
RETURN: Center point matrix,C.   

2) _Silhouette(X,Y,K)_: To check the efficacy of the produced result. We calculate the a,b and s values.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  a= distance of a point from all the same cluster points excluding itself.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b= mean intra cluster distance.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?s%3D%5Cbegin%7Bcases%7D%200%26%20%5Ctext%7B%20if%20%7D%20a%3Db%20%5C%5C%201-%5Cfrac%7Ba%7D%7Bb%7D%26%20%5Ctext%7B%20if%20%7D%20a%3Cb%20%5C%5C%20%5Cfrac%7Bb%7D%7Ba%7D-1%26%20%5Ctext%7B%20if%20%7D%20a%3Eb%20%5Cend%7Bcases%7D)  
An averaged Frobenius norm sum.The Frobenius norm is given by  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?%7C%7CA%7C%7C_F%20%3D%20%5B%5Csum_%7Bi%2Cj%7D%20abs%28a_%7Bi%2Cj%7D%29%5E2%5D%5E%7B1/2%7D)    
RETURN: The Silhouette score as an average of all the silhouette values.  

**PROCESS**:  
**Step 1**: Centroid Initialisation  
**Step 2**: Distance matrix calculation- Calculate the distance of each point with the initialised cluster points.  
**Step 3**: Cluster assignment- assigning each point with the label which has the minimum distance.  
**Step 4**: Update Centroid- Mean value of the points in a cluster.    
**Step 5**: Stopping criterion- Stop when the distance of previous and new center point matrix is zero.  

**RETURN**:   
>*y*: Predicted class label for the respective points.   
*c*: Centroids points for each cluster.  
