# MachinePy
A python library based on NumPy, SymPy, Matplotlib, SciPy.

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
Pranay Lawhatre and Prashant Lawhatre

License
----

MIT

[K-Nearest Neighbors]: <https://github.com/pvlawhatre/MachinePy/tree/master/Classification/KNN>
