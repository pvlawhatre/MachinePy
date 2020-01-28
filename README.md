# MachinePy
A python library based on NumPy, SymPy, Matplotlib.

## Introduction
The package has been written in for supervised, unsupervised and semi-supervised machine learning algorithms. All the major basic and advanced machine learning algorithms are written for clustering, rergression, classificaton, dimensionalty reduction problems. Aim of the project is purely educational and it was considered to be optimal while writing.

## Functionality
Currently, the release contain several models for the above stated functions. The structure of the proect is as follows:
* **Clustering**: [K-Means],[(K-means++)], [K-Median], [(K-Median++)], [Heirarchical], [Mean-Shift], [Fuzzy C-Mean],[Gaussian Mixture Models] and [Spectral Clustering]. 
* **Regression**: [Linear Regreession], [Linear Regresson L2 regularised], [Linear Regression L1 Regularised], [Linear regression L1,L2 Regularised], [MLE Linear Regression], [Bayesian Ridge Regression], [Gaussian Processes], [RANSAC], [Nadaraya Watson Regression], [Local Regression], [KNN Regression], [Perceptron/ADALINE Regression], [Chebyshev-FLNN], [Legendre-FLNN], [Laguerre-FLNN] and [Radial Basis Function Neural Net]. 
* **Classification**: [K-Nearest Neighbors], [Logistic Regression], [Logistic Regression L2 Regularised], [Logistic Regression L1 Regularised], [Logistic Regression L1 L2 Regularised], [Linear Discriminant Analysis], [Quadratic Discriminant Analysis], [Naive Bayes], [SVM], [Perceptron] and [RBF Neurl Network].

* **Dimensionality Reduction**: [Principal Component Analysis (PCA)], [Probabilistic PCA], [Random Projection], [classical Multi-Dimension Scaling(cMDS)], [LDA], [Kernel PCA], [Kernel LDA], [Isomap] and [Discriminant Neighborhood Embedding (DNE)].


* **Semi-Supervised Clustering**: [Constrained K-means], [Seed K-means] and [COP K-means].
* **Semi-Supervised Regression**:[Co-Training Regression].
* **Semi-Supervised Classification**:[Pseudo Labelling], [Cluster & Label], [Self Training] and [Co-Training].
* **Semi-Supervised Dimensionality Reduction**:[SSDR-M], [SSDR-CM & SSDR-CMU] and [SSDR-Manifold].

### Installation

Dillinger requires [Node.js](https://nodejs.org/) v4+ to run.

Install the dependencies and devDependencies and start the server.

```sh
$ cd dillinger
$ npm install -d
$ node app
```

For production environments...

```sh
$ npm install --production
$ NODE_ENV=production node app
```

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

First Tab:
```sh
$ node app
```

Second Tab:
```sh
$ gulp watch
```

(optional) Third:
```sh
$ karma test
```
```sh
cd dillinger
docker build -t joemccann/dillinger:${package.json.version} .
```
This will create the dillinger image and pull in the necessary dependencies. Be sure to swap out `${package.json.version}` with the actual version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on your host. In this example, we simply map port 8000 of the host to port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart="always" <youruser>/dillinger:${package.json.version}
```

Verify the deployment by navigating to your server address in your preferred browser.

```sh
127.0.0.1:8000
```
## Contributor
Prashant Lawhatre and Pranay Lawhatre

License
----

MIT

[Linear Regreession]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Linear%20Regression%20L1>  
[Linear Regresson L2 regularised]: <https://github.com/pvlawhatre/MachinePy/tree/master/Regression/LInear%20Regression%20L2>  
[Linear Regression L1 Regularised]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Linear%20Regression%20L1>  
[Linear regression L1,L2 Regularised]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Linear%20Regression%20L1L2>  
[MLE Linear Regression]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/MLE%20Linear%20Regression>  
[Bayesian Ridge Regression]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Bayesian%20Ridge%20Regression>  
[Gaussian Processes]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Gaussian%20Processes>  
[RANSAC]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/RANSAC>  
[Nadaraya Watson Regression]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Nadaraya-Watson%20Regression>  
[Local Regression]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Local%20Regression>  
[KNN Regression]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/KNN%20Regression>  
[Perceptron/ADALINE Regression]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/Perceptron_ADALINE%20Regression>   
[Radial Basis Function Neural Net]:<https://github.com/pvlawhatre/MachinePy/tree/master/Regression/RBF%20Neural%20Nets>  
[K-Nearest Neighbors]: <https://github.com/pvlawhatre/MachinePy/tree/master/Classification/KNN>  
[Logistic Regression]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/Logistic%20Regression>  
[Logistic Regression L2 Regularised]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/Logitic%20Regression%20L2>  
[Logistic Regression L1 Regularised]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/Logistic%20Regression%20L1>  
[Logistic Regression L1 L2 Regularised]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/Logistic%20Regression%20L1L2>  
[Linear Discriminant Analysis]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/LDA>  
[Quadratic Discriminant Analysis]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/QDA>  
[Naive Bayes]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/Naive%20Bayes>  
[SVM]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/SVM>  
[Perceptron]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/Perceptron>  
[RBF Neurl Network]:<https://github.com/pvlawhatre/MachinePy/tree/master/Classification/RBF%20Neural%20Net>  
[Principal Component Analysis (PCA)]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/PCA>  
[Probabilistic PCA]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/Probabilistic%20PCA>  
[Random Projection]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/Random%20Projection>  
[classical Multi-Dimension Scaling(cMDS)]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/cMDS>  
[LDA]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/LDA>  
[Kernel PCA]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/Kernel%20PCA>  
[Kernel LDA]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/Kernel%20LDA>  
[Isomap]:<https://github.com/pvlawhatre/MachinePy/tree/master/Dimensionality%20Redcuction/Isomap>  
[Discriminant Neighborhood Embedding (DNE)]:<>  
[Constrained K-means]:<>  
[Seed K-means]:<>  
[COP K-means]:<>  
[Co-Training Regression]:<>  
[Pseudo Labelling]:<>  
[Cluster & Label]:<>  
[Self Training]:<>  
[Co-Training]:<>  
[SSDR-M]:<>  
[SSDR-CM & SSDR-CMU]:<>  
[SSDR-Manifold]:<>  
[K-Means]:<>  
[(K-means++)]:<>  
[K-Median]:<>  
[(K-Median++)]:<>  
[Heirarchical]:<>  
[Mean-Shift]:<>  
[Fuzzy C-Mean]:<>  
[Gaussian Mixture Models]:<>  
[Spectral Clustering]:<>  
[Chebyshev-FLNN]:<>  
[Legendre-FLNN]:<>  
[Laguerre-FLNN]:<>  
