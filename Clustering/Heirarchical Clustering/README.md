# HEIRARCHICAL CLUSTERING

**SIGNATURE**:   
>_HCA(x, K)_

**DOCSTRING**:  
Perform heirarchical clustering with the given input training data and predict (y) label for clustering.  

**INPUT**:  
>*x*= Given the training data matrix in ndarray  
*k*= Mimimum K distance value.

**FUNCTION INSIDE THE CODE**:  
1) _sim(x,y)_ : To calculate the similarity matrix by Ward method which is, 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S%3D%5Cfrac%7B%5Cleft%20%7C%20%5Csum%20%5Cleft%20%7C%28x_%7Bi%7D-x%7Bj%7D%29%5E%7B2%7D%20%5Cright%20%7C%20%5Cright%20%7C%7D%7B%5Ctext%7BNumber%20of%20iterations%7D%7D) 
RETURN:  Ward distance among the given points.  

2) _low_mat(D)_: Calculates the position of the point which is closer in the cluster.  
RETURN: Index of that point in distance matrix.   

**PROCESS**:  
**Step 1**: Calculate the proximity of individual points and consider all the data points as individual clusters.  
**Step 2**: Similar clusters are merged together and formed as a single cluster.    
**Step 3**: Again calculate the proximity of new clusters and merge the similar clusters to form new clusters.  
**Step 4**: All the clusters are merged together and form a single cluster.     

**RETURN**:   
>*y*: Predicted class label for the respective points.   