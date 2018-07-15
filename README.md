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

<p align="center">![alt text](https://imgur.com/a79hAkQ.png "Step 1 formulas")</p>

C is the set of all centroids.

## Step 2

In this step we assign each input value to closest center. This is done by calculating Euclidean(L2) distance between the point and the each centroid.

<p align="center">![alt text](https://imgur.com/BadzRvI.png "Step 2 formulas")</p>

Where dist(.) is the Euclidean distance.

## Step 3

In this step, we find the new centroid by taking the average of all the points assigned to that cluster.

<p align="center">![alt text](https://imgur.com/IebqOfU.png "Step 3 formulas")</p>

S<sub>i</sub> is the set of all points assigned to the i<super>th</super> cluster.

