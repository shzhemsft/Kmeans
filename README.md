# Introduction

Clustering is a type of Unsupervised learning. This is very often used when you don’t have labeled data. K-Means Clustering is one of the popular clustering algorithm. The goal of this algorithm is to find groups(clusters) in the given data. In this post we will implement K-Means algorithm using Python from scratch. 

K-Means is a very simple algorithm which clusters the data into K number of clusters. The following image is an example of K-Means Clustering. 

![alt text](https://i.imgur.com/S65Sk9c.jpg "K-Means Clustering")

K-Means clustering is widely used for many applications such as: 
+ Image Segmentation
+ Clustering Gene Segementation Data
+ News Article Clustering
+ Clustering Languages
+ Species Clustering
+ Anomaly Detection

# Algorithmic walk-through

Our algorithm works as follows, assuming we have inputs x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>,…,x<sub>n</sub> and value of K. 

+ Step 1 - Pick K random points as cluster centers called centroids.
+ Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid.
+ Step 3 - Find new cluster center by taking the average of the assigned points.
+ Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.


![alt text](https://i.imgur.com/k4XcapI.gif "K-Means Clustering")

## Step 1

We randomly pick K cluster centers(centroids). Let’s assume these are c<sub>1</sub>,c<sub>2</sub>,…,c<sub>k</sub>, and we can say that: 

![alt text](https://imgur.com/a79hAkQ.png "Step 1 formulas")

C is the set of all centroids.

## Step 2

In this step we assign each input value to closest center. This is done by calculating Euclidean(L2) distance between the point and the each centroid.

![alt text](https://imgur.com/BadzRvI.png "Step 2 formulas")

Where dist(.) is the Euclidean distance.

## Step 3

In this step, we find the new centroid by taking the average of all the points assigned to that cluster.

![alt text](https://imgur.com/IebqOfU.png "Step 3 formulas")

S<sub>i</sub> is the set of all points assigned to the i<sup>th</sup> cluster.

## Step 4

In this step, we repeat step 2 and 3 until none of the cluster assignments change. That means until our clusters remain stable, we repeat the algorithm.

# Choosing the value of K

We often know the value of K. In that case we use the value of K. Else we use the **Elbow Method**.

![alt text](https://i.imgur.com/k3o6NxK.jpg "Choosing the value of k")

We run the algorithm for different values of K(say K = 10 to 1) and plot the K values against SSE(Sum of Squared Errors). And select the value of K for the elbow point as shown in the figure.

# Implementation with Python

The dataset we are gonna use has 3000 entries with 3 clusters. So we already know the value of K (i.e., K = 3).

We will start by importing and plotting the dataset.

```python
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('https://raw.githubusercontent.com/mubaris/friendly-fortnight/master/xclara.csv')
print("Input Data and Shape")
print(data.shape)
data.head()

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

plt.show()
```
![alt text](https://mubaris.com/files/images/output_3_1.png "Raw plot")

Now, we randomly initiate K cluster centers (centroids). Since K = 3, we initiate three centroids. 

```python
# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids")
print(C)

# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

plt.show()
```
![alt text](https://mubaris.com/files/images/output_6_1.png "raw plot with three centroids")

The three green stars are the centroids. Please note that your centroids will be located differently from the ones on the plot above because all centroids are initiated at random positions :) As long as there are three green stars, you are good to go! 

Now, we implement steps 2, 3, and 4 of the algorithm. At the end of the algorithm, there should be three clusters with properly positioned centroids. 

```python
# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

plt.show()
```

![alt text](https://mubaris.com/files/images/output_8_1.png "final result with K = 3")

If you run K-Means with wrong values of K, you will get completely misleading clusters. For example, if you run K-Means on this with values 2, 4, 5 and 6, you will get the following clusters. 

![alt text](https://i.imgur.com/chdUE0r.png "final result with K = 2")

![alt text](https://i.imgur.com/I9J0y2A.png "final result with K = 4")

![alt text](https://i.imgur.com/cly9t5S.png "final result with K = 5")

![alt text](https://i.imgur.com/z2N2oSV.png "final result with K = 6")


# K-means clustering with Python sklearn

You just implemented your own K-means clustering algorithm *from scratch*, and you may be wondering if Python has built-in support for it? The answer is yes :) Details below. 

We will first generate a new dataset using **make_blobs** function.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize'] = (16, 9)

# Creating a sample dataset with 4 clusters
X, y = make_blobs(n_samples=800, n_features=3, centers=4)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])

plt.show()
```
![alt text](https://mubaris.com/files/images/output_14_1.png "k-means with python library")

```python
# Initializing KMeans
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

plt.show()
```
![alt text](https://mubaris.com/files/images/output_16_1.png "k-means with python library")

In the above image, you can see 4 clusters and their centroids as stars. scikit-learn approach is very simple and concise.

# The limitations of k-means clustering

K-means is a widely used method in cluster analysis. Technically, this method does NOT require ANY assumptions, i.e., give me a dataset and a pre-specified number of clusters, k, and I can just apply this algorithm which minimizes the sum of squared errors (SSE), the within cluster squared error. However, there are very few "free lunches" in our life. The easy-of-use of the algorithm comes with several limitations/drawbacks, as described below. 

## Limitation 1: the trouble with closed frontiers

![alt text](https://i.stack.imgur.com/g5Jb8.png "k-means fails for spherical frontier")

![alt text](https://i.stack.imgur.com/SlpL1.png "k-means fails for spherical frontier")

But a quick treatment on the original dataset solves the problem. Can you see it? Hint: polar coordinates. 

![alt text](https://i.stack.imgur.com/0sUph.png "k-means with polar coordinates")


## Limitation 2: the trouble with non-uniform data density

![alt text](https://i.stack.imgur.com/WiH4T.png "k-means with un-even input")

![alt text](https://i.stack.imgur.com/zAI1g.png "k-means with un-even input")


# Now let's take a step back...

We have covered classification, regression, and clustering in this class. Can you compare and contrast these three data science techniques? 

## Regression

regression analysis is a set of statistical processes for estimating the relationships among variables. It includes many techniques for modeling and analyzing several variables, when the focus is on the relationship between a dependent variable and one or more independent variables (or 'predictors'). More specifically, regression analysis helps one understand how the typical value of the dependent variable changes when any one of the independent variables is varied, while the other independent variables are held fixed.

## Classification

Classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known. Examples are assigning a given email to the "spam" or "non-spam" class, and assigning a diagnosis to a given patient based on observed characteristics of the patient (gender, blood pressure, presence or absence of certain symptoms, etc.). In the terminology of machine learning, classification is considered an instance of supervised learning, i.e. learning where a training set of correctly identified observations is available. The corresponding unsupervised procedure is known as clustering, and involves grouping data into categories based on some measure of inherent similarity or distance.

## CLustering 

Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data mining, and a common technique for statistical data analysis, used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compression, and computer graphics.

![alt text](https://i.stack.imgur.com/sxEi9.jpg "clustering vs. classification")

![alt text](http://cdn.differencebetween.net/wp-content/uploads/2018/01/Differences-Between-Supervised-Learning-and-Unsupervised-Learning.png "comparisons")













