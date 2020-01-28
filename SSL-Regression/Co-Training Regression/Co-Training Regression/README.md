# CO-TRAINING REGRESSION

**SIGNATURE**:   
>_coreg(x_lbl,y_lbl,x_unlbl,K,T,P1,P2)_  

**DOCSTRING**:  
Performs co-training regressoin over the given data matrix for regression.  

**INPUT**:  
>*x_lbl*= Given the labeled input data matrix in ndarray  
*y_lbl*= Given the labeled output data matrix in ndarray  
*x_unlbl*: Given the unlabeled data matrix in ndarray
*k*= Number of nearest neighborsk  
*T*: Maximum number of learning iterations  
*P1*:Distance order for training data (x_lbl[:split])
*P2*:Disance order for testing dat (x_lbl[split:])  

**FUNCTIONS INSIDE THE CODE**  
1) _KNN_reg(x_train,y_train,x_test,K,P)_: Performs KNN regression.
2) _nbh(xp,k,S)_: Calculates the Neighbourhood of the points and return index of top *K* nearest points.  

**PROCESS**:  
**Step 1**: Randomly picking examples from unlabeled set.  
**Step 2**: Peform KNN for two regressor one each for the half of the labeled dataset.  
**Step 3**: For each point choosen randomly from the unlabeled data points is tested by both the regressors and calculating its neighbours   
**Step 4**: A new regressor is formed which has the given input data plus the new test data point. Square error for both the orignal and the newly formed regressor is substracted. Repeat till T rounds.    
**Step 5**: If the two regressor or each case is not similar then, the maximum random data point and its label is removed from the unlabeled set of randomly picked point.  
**Step 6**: These points and its label is appended in the two halves data points regressor and replenish the unlabeled data popints by randomly picking examples from the original unlabeled data points. Output regressor is the average of two.  

**RETURN**:   
>*mean(P)*:  Average of the two regressors.
 