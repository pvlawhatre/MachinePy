# RBF NEURAL NETS

**SIGNATURE**: 
>RBFN_reg(X_train,y_train,X_test,no_h)

**DOCSTRING**:
Perform RBF neural nets with the given input training data and predict ( x_test data matrix) result.
**INPUT**:
>X_train= Given the training data matrix in ndarray
y_train= Given the training label array in ndarray
X_test= Given the test data matrix in ndarray
no_h=Number of neuron in the hidden layer.

**FUNCTION INSIDE THE CODE**:
1) *kpp(X,K)*: Performs the K-mean++ for centre intialisation.
RETURN: *c*, ndarray of shape (k,dim)

**PROCESS**:
**Step 1**: Centre of Gaussians by Kmean++ (*kpp()*)
**Step 2**: Hidden layer activation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?S%3Dexp%5Cleft%20%28%20-%5Cfrac%7B%28xtrain-C_%7Bi%7D%29%5E%7B2%7D%7D%7B2%7D%20%5Cright%20%29)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Here, *Ci* indicates that each centres is substracted from one instance of x_train value.

**Step 3**: Closed form solution
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?W%3D%28S%5E%7B-1%7D%5Cbullet%20ytrain%29%5E%7BT%7D)
**Step 4**:Prediction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?ytest%3DS%5Cbullet%20W%5E%7BT%7D%2C%5C%3B%20where%5C%3B%20S%5C%3B%20is%5C%3B%20S%28xtest%2CC%29)

**RETURN**: 
>y_test: test prediction with shape similar to y_train.
