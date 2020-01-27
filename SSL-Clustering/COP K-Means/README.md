# SSL- SEED K-MEANS

**SIGNATURE**:   
>_cop_kmean(X,K,must_lnk,cannot_lnk)_    

**DOCSTRING**:  
Perform Cop K-means with the given input training data and predict ( x_test data matrix) result for clustering.  

**INPUT**:  
>*X*= Given the training data matrix in ndarray  
*K*= K value for Kmean++  
*must_link*= Data points that are to be together.
*cannot_link*= Data points that cannot be together.

**FUNCTIONS INSIDE THE CODE**:  
1) _kpp(X,K)_: Performs Kmean++. for center initialisation.  

**PROCESS**:  
**Step 1**: Assign instances from dataset in must link and cannot link group  
**Step 2**: K clusters initialisation  
**Step 3**: For each point in the labelled dataset assigned it to the closest cluster such that it doesn't violet the constraints. If no such cluster exisit, fail.
**Step 4**: Repeat until convergence.  

**RETURN**:   
>*X*: Test data matrix.  
*Y*: Test prediction unlabelled data.