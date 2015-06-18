clustering become important as it goes. However, the clustering algorithm takes O(n<sup>2</sup>) ~ O(n<sup>3</sup>) generally. this make diffcult to cluster a large number of datapoints. BIRCH is one of the scalable clustering algorithm that can performs in nearly linear time. The main idea of BIRCH is that some nearby datapoints can be grouped into one representative cluster. after collecting these summarized clusters instead of datapoints themselves, clustering is performed on the smaller number of summarized clusters.

based on the paper:
  * "BIRCH: An Efficient Data Clustering Method for very Large Databases"

code references:
  * http://www.cs.wisc.edu/~zhang/birch.html // Authors' but unavailable now
  * http://roberto.perdisci.com/projects/jbirch // JBIRCH

description in the wikipedia:
  * http://en.wikipedia.org/wiki/BIRCH_%28data_clustering%29