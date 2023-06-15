# CS205_PROJECT 2
### Name: Dhananjay Gavade
### SID: 862395404
### email: dgava001@ucr.edu
### Name: Ajay Wayase
### SID: 862394912
### email: awaya001@ucr.edu

#### Code Exection Steps:

1. Execute feature_search.py
2. Specify file. (for real world dataset specify real_world_rice_data.csv)
3. Use the notebook Project_2_AI.ipynb
#### Requirements:
1. scipy - for reading real world dataset which is in arff format (command- pip install scipy)
2. scikitlearn - for label encoding (command- !pip3 install scikit-learn)
3. numpy -  for vectorization of nearest neighbour function (command- !pip3 install numpy)

#### Algorithms:
##### Nearest Neighbour: 
Nearest Neighbor is an instance-based classification algorithm. In this algorithm model makes decisions based on similarity between new input and training examples. The idea behind the nearest neighbor algorithm is that we should find nearest neighbors for new inputs. The algorithm considers that the points that are close to each other in the feature space will have the same labels. So the algorithm finds the nearest neighbor for the new input point and assigns that neighbor’s class to the new input point.
##### Forward Selection:
In forward selection we first select each feature individually and calculate the accuracy. This is the locally optimal feature. We will keep on adding features in this feature set. At the end we will get a subset with all optimal features. 

##### Backward Elimination:
In backward elimination we will consider all the features at the start and then we will eliminate one-one irrelevant features greedily from the subset of features. 


