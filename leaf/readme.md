# A Study of Rhododendron Leaf Shape Variability in the Russian Far East

This is main heading. Common description will be here.


##Materials and Methods


###Dimensionality reduction

We used PCA as a main dimensionality reduction technique. Initially, leaf shape was characterized by angles of all possible   
triangles formed on a set of landmark points. The set of landmark points was chosen as follows:
   
<p align="center">
<img src="https://raw.githubusercontent.com/scidam/leafs/master/leaf/imgs/landmarks.png">
</p>

It was 168 different angles, so we had unsupervised classification problem in a quite high dimensional space. 

We obtained the following PCA-weights (explained variance ratios): 

>  [ 0.03370089  0.03330639  0.03231536  0.03134169  0.03060391  0.03018004
>  0.02981616  0.02468735  0.02397441  0.02308131  0.02283795  0.02246768
>  0.02209255  0.02165169  0.02143578  0.02100789  0.02092459  0.02063437
>  0.0204008   0.02008309  0.01966714  0.01928566  0.018771    0.01864633
>  0.01827244  0.01766271  0.01703517  0.00586928  0.00577847  0.00563653
>  0.0055381   0.00550945  0.00545177  0.00533217  0.00528444  0.00524421
>  0.00512102  0.0051023   0.00503153  0.00496849  0.00487448  0.00481758
>  0.0047192   0.00468047  0.00461325  0.00458282  0.00453561  0.00441768
>  0.00386305  0.00381909]
 
From these values one can see that first 27 values are quite bigger than last. So, we can use first 27 principal components
to describe leaf  shape morphology. These 27 principal components explain about 63.5% of total variance. 
Increasing the number of components to 50, allow us to increase this value to 75%.
Currently, we decided to stop dimensionality reduction at 50 principal components.


![K-means with k = 4](https://raw.githubusercontent.com/scidam/leafs/master/leaf/imgs/kmeans4.png)

<p align="center"> K-means clustering with k=4</p>


![Spatial distribution](https://raw.githubusercontent.com/scidam/leafs/master/leaf/imgs/distrib.png)

<p align="center"> Spatial distribution of the data points</p>


![Spatial distribution](https://raw.githubusercontent.com/scidam/leafs/master/leaf/imgs/meanshps.png)

<p align="center">Mean leaf shapes of clusters</p>


##Conclusion 
Common conclusion on the causes of the variability.
