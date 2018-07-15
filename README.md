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

![alt text](https://i.imgur.com/k4XcapI.gif "Step 1 formulas")

