# Customer-Segmentation
An analysis of customer data using unsupervised learning methods such as K-Means, Hierarchial Clustering and Principal Component Analysis

- [**Table of Contents**](#customer-segmentation)
  - [Introduction](#introduction)
  - [Setting up the Project](#setting-up-the-project)
    - [Importing Data and Libraries](#importing-data-and-libraries)
  - [Part 1 - Exploratory Data Analysis](#part-1---exploratory-data-analysis)
    - [Some of the interesting things found:](#some-of-the-interesting-things-found)
  - [Part 2 - Data Pre-processing](#part-2---data-pre-processing)
    - [Dealing with categorical data columns](#dealing-with-categorical-data-columns)
  - [Part 3 - Build and Evaluate Models](#part-3---build-and-evaluate-models)
    - [K-Means Clustering](#k-means-clustering)
      - [Determining the Ideal number of Clusters in the dataset](#determining-the-ideal-number-of-clusters-in-the-dataset)
    - [Hierarchial Clustering](#hierarchial-clustering)
    - [Principal Component Anaylsis](#principal-component-anaylsis)
  - [Conclusion](#conclusion)
  - [References](#references)

 ## Introduction

 Customer Segmentation is the practice of dividing the customers of a company ito different groups that possess similarities in each group. The objective of segmenting the customers is to ascertain how to relate to customers in each segment so that we can maximize the value of each customer to the business.
Ideally this allows us to tailor our appraoch to each customer group based on their interests, demographic profile or even preferered method of communication and interaction.
 
In this notebook we seek to identify customer segments so that we can deliver insights about our data through clusterig
and means. This will be used to drive business decisions in tailoring approaches to the different customer bases/types.
The data was obtained from [Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) and it includes features such as Customer ID, Age, Gender, Annual Income and Spending Score (which was assigned based on the internal metrics of the company. 


 ## Setting up the Project

 ### Importing Data and Libraries
1. Load libraries and obtain data from [Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).
2. Store data in an appropriate dataframe.
3. Set a random seed for reproducibility.



## Part 1 - Exploratory Data Analysis
In this section we will investigate the data, cleaning and modifying where necessary to make it easier to manipulate and see what insights we can find. <br>
Also you can look at the output file of pandas- profiling [here](https://htmlpreview.github.io/?https://github.com/OjeWilliams/Customer-Segmentation/blob/main/data/output.html) and you can download the file [here](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/data/output.html) (it is interactive once downloaded)

Below we can see the first 10 entries of the dataset as well as seeing that there are no missing values. 
![](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/images/initialcheck.png) <br> <br>

### Some of the interesting things found:
- Gender percentages
![](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/images/genders.png) <br> <br>

- Distribution Plots
![](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/images/DistributionPlots.png) <br> <br>

- Mean Spending Score
![](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/images/meanscore.png) <br> <br>

- Median Annual Income
![](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/images/medianincome.png) <br> <br>

The full break down of everthing that was explored can be found [here](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/code/Customer%20Segmentation.ipynb) in the notebook.


<div style="page-break-after: always"></div>

## Part 2 - Data Pre-processing
In this this section we deal preparing the data for it to be used in our models.

### Dealing with categorical data columns
Here we have two issues to address, we have to transform our categorical data values into numerical values and then we have to scale our data. 
Categorical variables are those that are labels rather than numeric values such as _'color'_ or a _'place/location'_ or in our case _gender_, i.e male or female. We need to address these categorical data values as some of our methods below will not work well with them and hence must be converted to numerical values.
 Scaling our data means that we are normalizing the range of the features of our data as a means to combat the large range of values in our raw data. 

 We addressed both tasks as seen below:
 ![](https://github.com/OjeWilliams/Customer-Segmentation/blob/main/images/transformed.png)


## Part 3 - Build and Evaluate Models
### K-Means Clustering
K-Means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. k-means clustering minimizes within-cluster variances (squared Euclidean distances), but not regular Euclidean distances, which would be the more difficult Weber problem: the mean optimizes squared errors, whereas only the geometric median minimizes Euclidean distances. For instance, better Euclidean solutions can be found using k-medians and k-medoids.

#### Determining the Ideal number of Clusters in the dataset
When determining the number of clusters to use one of the important methods is using what is known as the **_Elbow Curve Method_** This method involves plotting the explained variance( Sum of Squared Distance) vs the Number of clusters (k) and choosing the number of clusters based on where the 'elbow' occurs. The elbow is the cutoff point at which increasing the number of clusters is no longer worth it. We also utilized the **_Silhouette Score_** to help with selection.
![](ElbowPlot.png)

I had a choice between two possible cluster numbers and I choice to build models based on 6 rather than 3 clusters.

- 6 Cluster Comparison Plots
![](KM6-ClusterComparison.png) <br> <br>

- 3D plot of 6 Cluster Model
![](KM6-Cluster.png)

- Cluster Summary
![](ClusterSummary.png) <br> 


### Hierarchial Clustering
In data mining and statistics, hierarchical clustering (also called hierarchical cluster analysis or HCA) is a method of cluster analysis which seeks to build a hierarchy of clusters. In other words it is an algorithm that groups similar objects into groups called clusters. There are mainly two types

Agglomerative: This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
Divisive: This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In this notebook we used the Agglomerative approach and used a dendrogram to visualize results

- Total dendrogram
![](TotalDendrogram.png) <br> <br>

- Partial Denmdrogram (suggesting 7 clusters)
![](TruncDendrogram.png) <br> <br>

### Principal Component Anaylsis

Principal component analysis (PCA), is one of the most popular unsupervised machine learning techniques. It is primarily used as a dimensionality reduction approach with applications in data visualization and feature extraction, allowing for increased interpretability while minimizing information loss.
In dimensionality reduction, the goal of the inferred model is to represent the input data with the fewest number of dimensions, while still retaining information such as the variability in the data that is relevant to investigation. New component generated are combinations of proportions of existing features and these components explain the maximum variance in the model.

- PCA Summary
![](PCA-Summary.png)


## Conclusion
From our investigation into this data set using unsupervised maching learning methods we have found that: <br>  <br>
- **K-Means Clustering** initially suggested that we can segment our customers into 3 or 6 groups. <br> <br>
- **Hierarchial Clustering** suggested that we can segment our customers into 7 groups. <br>  <br>
- **Principal Component Analysis** returned 2 components that explained 67% percent of the variance in our data. Using          these two components we were able to build a better K-Means model that suggested our customers could be put into 4 groups.

From the 6 Cluster model I was able to create coteries to describe each group in detail while also providing insights into what might appeal to each group. This can be found in the notebook.

## References
- https://towardsdatascience.com/customer-segmentation-with-python-31dca5d5bdad?source=user_profile
- https://www.kaggle.com/datark1/customers-clustering-k-means-dbscan-and-ap
