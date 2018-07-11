# Machine-Learning-Models
This repository contains implementation of basic ML algorithms Models in Python without use of any frameworks.

### Algorithms

#### [Neural Network](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/digitClassifier.py)
Supervised learning using a Neural Network to predict a function that classisfies MNIST digits.

Neural network structure: The neural network takes in (28x28) 784 inputs, one hidden layer with `n` hidden units (where n is a parameter that can be changed), and 10 output units`(0-9)`. The hidden and output units uses [sigmoid activation function](https://en.wikipedia.org/wiki/Sigmoid_function). The network is fully connected —that is, every input unit connects to every hidden unit, and every hidden unit connects to every output unit. Every hidden and output unit also has a weighted connection from a bias unit, whose value is set to 1.

##### Parameters of the network:

  Number of neurons in the hidden layer - [n](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/a5163da699f9cabe7b109b084ab3d96fccaef532/digitClassifier.py#L32) = 100
  
  Number of epochs - [epochs](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/a5163da699f9cabe7b109b084ab3d96fccaef532/digitClassifier.py#L8) = 50
  
  Learining rate - [eta](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/a5163da699f9cabe7b109b084ab3d96fccaef532/digitClassifier.py#L11) = 0.1
  
  Momentum - [alpha](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/a5163da699f9cabe7b109b084ab3d96fccaef532/digitClassifier.py#L14) = 0.9
  

##### Accuracy vs Parameters

η | α | n | Training Accuracy| Test Accuracy |
--|---|---|------------------|---------------|
0.1|0.9|20|[96%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-1-n-20.png)|[93%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-1-n-20.png)|
0.1|0.9|50|[98%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-1-n-50.png)|[96.8%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-1-n-50.png)|
0.1|0.9|100|[99.6%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-1-n-100.png)|[96.85%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-1-n-100.png)|
0.1|0|100|[99.6%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-2-alpha-0.png)|[97.7%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-2-alpha-0.png)|
0.1|0.25|100|[99.6%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-2-alpha-0.25.png)|[97.7%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-2-alpha-0.25.png)|
0.1|0.5|100|[99.6%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-2-alpha-0.5.png)|[97.6%](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/plots/exp-2-alpha-0.5.png)|

#### [k-Means Clustering](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/kMeans.py)
Unsupervised learning with [k-Means](https://en.wikipedia.org/wiki/K-means_clustering) clustering using [EM algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
##### Parameters
`k - number of clusters`

Input is 1500 data points(2d) taken from random Gaussian distribution. Although it can accept any 2d data as input.
Plots and log-likelihood for different values of `k` can be found [here](https://github.com/rishab-pdx/Machine-Learning-Algorithms/tree/master/km-plots)

##### Clusters
k = 2
![k=2](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/km-plots/km-2-l.png)
k = 3
![k=3](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/km-plots/km-3-l.png)
k = 4
![k=4](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/km-plots/km-4-l.png)
k = 5
![k=5](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/km-plots/km-5-l.png)
3. 
#### [Gaussian Mixture Models](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/gaussianMM.py)
Unsupervised learning with [GMM](https://brilliant.org/wiki/gaussian-mixture-model/) using [EM algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
##### Parameters
`k - number of clusters`

Input is 1500 data points(2d) taken from random Gaussian distribution. Although it can accept any 2d data as input.
Plots and log-likelihood for different values of `k` can be found [here](https://github.com/rishab-pdx/Machine-Learning-Algorithms/tree/master/km-plots). It makes use of [k-Means Clustering](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/kMeans.py) to get `k` clusters and then run Gaussian model and predicts a Gaussian around the data.
##### Gaussians
k = 2
![k=2](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/gmm-plots/gmm-k-2-r-max-l-2.png)
k = 3
![k=3](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/gmm-plots/gmm-k-3-r-max-l.png)
k = 4
![k=2](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/gmm-plots/gmm-k-4-r-max-l.png)
k = 5
![k=2](https://github.com/rishab-pdx/Machine-Learning-Algorithms/blob/master/gmm-plots/gmm-k-5-r-max-l.png)

#### Dependencies
1. Python
2. Matplotlib
