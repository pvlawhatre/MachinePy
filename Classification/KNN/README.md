# K-NEAREST NEIGHBOURS (KNN)

**SIGNATURE**: 
>KNN(x_train,y_train,x_test,k,*arg)

**DOCSTRING**:

Perform KNN with the given input training data and predict ( x_test data matrix) result for classification.

**INPUT**:
>x_train= Given the training data matrix in ndarray
y_train= Given the training label array in ndarray
x_test= Given the test data matrix in ndarray
k= 'k' value.
*args= 1) **'cityblock'** : Performs cityblock distance. 2) '**minkowsky**': Perfrom Minkowsky distance. 3) '**hamming**': Perform Hamming distance.
NOTE: If nothing is provided DFAULT Euclidean distance will be performed.

**PROCESS**:

**Step 1**: Calculating the distances of each point from each 'k' centres.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?D%3D%5Csum_%7Bi%5Cepsilon%20n%7D%5Cleft%20%7Cu_%7Bi%7D-v_%7Bi%7D%20%5Cright%20%7C%2C%5C%3B%20%28Cityblock%29)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?D%3D%5Csum_%7Bi%5Cepsilon%20n%7D%5Cleft%20%28%5Cleft%20%7Cu_%7Bi%7D-v_%7Bi%7D%20%5Cright%20%7C%5E%7B2%7D%20%5Cright%20%29%5E%7B1/2%7D%2C%5C%3B%20%28Minkowsky%29)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?D%3D%5Cfrac%7Bc%7D%7Bn%7D%2C%5C%3B%20%28Hamming%29%20where%5C%3B%20c%5C%3B%20is%5C%3B%20the%5C%3B%20number%5C%3B%20of%5C%3B%20occurence%5C%3B%20of%5C%3B%201%5C%3B%20in%5C%3B%20XOR)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?D%3D%5Csum_%7Bi%5Cepsilon%20n%7D%5Cleft%20%28%20%5Cleft%20%7C%20u_%7Bi%7D-v_%7Bi%7D%20%5Cright%20%7C%5E%7B2%7D%20%5Cright%20%29%5E%7B1/2%7D)

**Step 2**: Selecting the label with most common one in K neighbours.

**RETURN**: 
>y_test: test prediction with shape similar to y_train.
