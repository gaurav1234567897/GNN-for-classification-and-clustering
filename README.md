# GNN-for-classification-and-clustering
GNN for node classification and clustering for community detection
Node classification using Kipf's semisupervised classification algorithm.

In this project I have employed the gumbel softmax
approach to graph network clustering. 

The experimental findings on specific graph datasets are used to compare with
spectral clustering. We do a series of experiments on our
graph clustering algorithm, using dataset: Zachary karate
club.
Graph clustering, aiming to partition a graph into several
densely connected disjoint communities or groups.

I. INTRODUCTION
Depending on how well the nodes of the graph are
connected, we detect the community structure in the
graph network, and group identification has significant
consequences for revealing the structure of human social
network.
Spectral clustering uses eigen vector approach.
Here, we will be using gumbel sortmax approach to select
and extract the features for the Graph neural network to
detect community structure in the graph datasets using deep
learning approach.

II. BACKGROUND AND RELATED WORK
A. Community Detection Algorithms
The algorithm proposed starts with all nodes as individual
communities and merge them iteratively to optimize the
function ’modularity’ to define communities that have many
numbers of edges within them and few between them.
Spectral method based on the modularity matrix’s is used to
optimize the modularity metric measure.
B. Gumbel Softmax Approach on Feature Selection
The method used in the experiment is defined as below:
Graph − Cluster(A) = Softmax(WCt AWC) (5)
In the Equation 5, ’A’ indicates the Adjacency matrix of the
undirected graph G and ’WC’ indicates the Gumbel cluster
weight matrix.
In general, let An×n be the adjacency matrix where ’n’
represents the total number of nodes in the graph dataset. The
size of the matrix WC is n × k, where k indicates the number of
clusters. When we perform WCt AWC operation, we obtain a
matrix of the size k×k and let us call this matrix as Rk×k. The
resultant Rk×k matrix shows the strength of the cluster where
the primary diagonal shows the strength of data points within
cluster groups, and other elements of the matrix R provide
details on the strength of data points between different cluster
groups. Then, we apply softmax function on the obtained
matrix R to express our inputs as a discrete probability
distribution. Mathematically, this is defined as follows:
Now, consider the trained Gumbel cluster weight
matrix(Wc), which is of the size n × k. Here, each row is a graph
node and will sum up to 1, and the index of the maximum row
value is the cluster to which the row node belongs. For
example, let us consider k = 2, i.e., we are trying to cluster the
dataset into 2 cluster groups. Here, we assume the first row
data as [0.9 0.1], where 0.9 is at the 0th index, and 0.1 is at the
1st index. Looking at the index of the maximum value in the row
vector, one can easily say that the graph node belongs to
cluster 0. If the second-row assumed data is [0.29 0.71], where
0.29 is at the 0th index, and 0.71 is at the 1st index. Looking at
the index of the maximum value in the row vector, one can
easily say that the graph node belongs to cluster 1. Likewise,
we look into all the rows and then cluster all the nodes of the
graph dataset into the cluster group. The row data values
indicate the influence of the graph node towards the cluster
group.
The result obtained from the equation 5 is compared with
the loss function. The loss function for our experiment used is
the identity matrix(Ik×k) where each diagonal element
represents the cluster group.
Using the gumbel softmax function and the method
proposed, we find the community cluster of the graph dataset
nodes.

IV. EXPERIMENT RESULTS
A. Datasets
1) Human Social Networks: Human social networks are
real-world social networks among human beings. The links are
offline, and not from a social network.
