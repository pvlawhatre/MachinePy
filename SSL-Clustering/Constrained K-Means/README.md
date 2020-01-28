# SSL- CONSTRAINED K-MEANS

**SIGNATURE**:   
>_constrained_kmeans(x_lbl,y_lbl,x_unlbl)_    

**DOCSTRING**:  
Perform constrained K-means with the given input training data and predict ( x_test data matrix) result for clustering.  

**INPUT**:  
>*x_lbl*= Given the training data matrix in ndarray  
*y_lbl*= Given the training label array in ndarray  
*x_unlbl*= Given the test data matrix in ndarray  

**PROCESS**:  
**Step 1**: Centre initialisation to the mean of every cluster.  
**Step 2**: Points assignment. Calculating the distance of every unlabelled data points to the center of clusters.  
**Step 3**: Output of unlabelled data points will be the shortest distance points for a particular cluster.  
**Step 4**: New centers are assigned after calculating the mean because of the newly addded data points. Repeat untill convergence.  

**RETURN**:   
>*y_unlbl*: Test prediction for first half of trianing dataset.
