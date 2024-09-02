# # Module 4 - Programming Assignment - k Nearest Neighbor
import random
from typing import List, Tuple

# There are 1,030 observations and each observation has 8 measurements. The data dictionary for this data set tells us the definitions of the individual variables (columns/indices):
# 
# | Index | Variable | Definition |
# |-------|----------|------------|
# | 0     | cement   | kg in a cubic meter mixture |
# | 1     | slag     | kg in a cubic meter mixture |
# | 2     | ash      | kg in a cubic meter mixture |
# | 3     | water    | kg in a cubic meter mixture |
# | 4     | superplasticizer | kg in a cubic meter mixture |
# | 5     | coarse aggregate | kg in a cubic meter mixture |
# | 6     | fine aggregate | kg in a cubic meter mixture |
# | 7     | age | days |
# | 8     | concrete compressive strength | MPa |
# 
# The target ("y") variable is a Index 8, concrete compressive strength in (Mega?) [Pascals](https://en.wikipedia.org/wiki/Pascal_(unit)).


# ## Load the Data
# The function `parse_data` loads the data from the specified file and returns a List of Lists. The outer List is the data set and each element (List) is a specific observation. Each value of an observation is for a particular measurement. This is what we mean by "tidy" data.
# The function also returns the *shuffled* data because the data might have been collected in a particular order that *might* bias training.

def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data

# ## Get statistics on the Data
# The function `get_min_max_mean` takes a List of Lists and returns a List of Tuples. 
# Each Tuple contains the minimum, maximum, and mean of the values for a particular feature.

def get_min_max_mean(data: List[List]) -> List[Tuple[float, float]]:
    min_max_mean = []
    for i in range(len(data[0])):
        col_values = [row[i] for row in data]
        min_max_mean.append((min(col_values), max(col_values), sum(col_values)/len(col_values)))
    return min_max_mean

# ## Normalize the Data
# The function `normalize` takes a List of Lists for dataset and a List of Tuples which holds min, max, and mean of each feature.
# The function returns a Tuple of Lists that holds the normalized data using min-max normalization and the mean of each feature.

def normalize(data: List[List], min_max_mean: List[Tuple[float, float, float]]) -> Tuple[List[List], List[List]]:
    normalized_data = []
    num_features = len(data[0])
    for i in range(num_features):
        # extract values for feature from all examples in data
        feature = [example[i] for example in data]
        # get the min and max of the feature in the database
        min_feature = min_max_mean[i][0]
        max_feature = min_max_mean[i][1]
        # normalize the feature by min-max normalization
        normalized_feature = [(value - min_feature) / (max_feature - min_feature) for value in feature]
        normalized_data.append(normalized_feature)
    # transpose the normalized data to have same format as the original data
    normalized_data = list(map(list, zip(*normalized_data)))
    return normalized_data


# ## Euclidian Distance
# `euclidian_distance` calculates the euclidian distance between two points in d dimentions, where d is equal to the number of independent features for each data point. 
# The distance is used as a metric by the knn algorithm to determine the k closest data points to the query point and uses the labels of those k neighbours to predict the label of the query point. 
# 
# **Used by**: [knn](#knn)
# 
# * **p1** List[float]: list containing features and labels of a data point instance.
# * **p2** List[float]: list containing features and labels of a data point instance.
# 
# **returns** float: distance between the two points. 

def euclidian_distance(p1: List, p2: List) -> float:
    return sum((p1[i] - p2[i])**2 for i in range(len(p2)-1))**0.5


# ## Predict
# `predict` performs weighted voting amongst the k neighbours of a query point to assign a numerical target label to the query point. The function works by calculating the weighted sum of the target numerical labels of k neighbours normalized by the sum of weights. The output is a regression target value assigend to the query point. The weight is proportional to the inverse of the distance between a neighbour and the query point. So the closer the point is to the query point, the higher its vote counts. 
# 
# **Used by**: [knn](#knn)
# 
# * **k_nearest_neighbors** List[Tuple[float, List]]: list of tuples where each tuple contains the euclidian distance from point to query and the features and target label of the point. 
# 
# **returns** float: predicted numerical target label for query point.  

def predict(k_nearest_neighbors: List[Tuple[float, List]]) -> float:
    weighted_sum_of_labels = 0
    sum_of_weights = 0
    for neighbor in k_nearest_neighbors:
        # weight is proportional to inverse of distance 
        weight = 1 / (1 + neighbor[0])
        # weight x target value
        weighted_sum_of_labels += weight * neighbor[1][-1]
        # normalization factor
        sum_of_weights += weight
    return weighted_sum_of_labels / sum_of_weights


# ## K Nearest Neighbour
# `knn` is an implementation of the k nearest neighbour algorithm which is a type of unsupervised machine learning algorithm used to perform either regression or classification. It is a lazy method because no explicit model is built. Rather, the learning takes place at query time in which the algorithm calculates the distance between all points in the training dataset to the query datapoint. Then, the algorithm sorts the points in the training set based on their relative distance to the query point, and picks the k nearest neighbours to perform regression or classification. The idea is that points that lie near each other should be similar to each other and hence should have similar target labels (either numeric labels or categorical labels). 
# 
# In our case, the knn algorithm is used to perform regression becuase we are trying to estimate the compressive strength of concrete in MPa (which is a continuous feature) given several parameters of the concrete composition and aging. The regression is performed by doing a weighted sum of the k nearest neighbours where the weight is inversely proportional to the distance between the train data points and the query point. The closer is the point to the query point, the more weight it target label receives. 
# 
# **Used by**: [cross_validate](#cross_validate)
# 
# * **data** List[List]: a list of lists where the outer lists are the data instances in the training dataset and each data instance is a list of features and a target label. 
# * **query** List[float]: a single data point instance that has features and a target label. c
# * **k** int: hyperparameter representing the number of neighbours around the query point to consider for predictions. 
# 
# **returns** float: predicted numerical target label for query point. 

def knn(data: List[List], query: List[float], k: int) -> float:
    distances = []
    for example in data:
        distances.append((euclidian_distance(example, query), example))
        # sort by distance and get the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    return k_nearest_neighbors

