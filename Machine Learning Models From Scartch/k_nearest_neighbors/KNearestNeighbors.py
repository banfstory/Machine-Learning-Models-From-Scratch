import numpy as np
from math import sqrt
from collections import Counter
import warnings

class KNearestNeighbors:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    # this will fit the model with the given data
    def fit(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        self.data_groups = {}
        # put each row into label groups
        for i in range(0, len(Y)):
            if(Y[i] not in self.data_groups):
                self.data_groups[Y[i]] = []
            self.data_groups[Y[i]].append(X[i])

    # determine the accuracy of the model based off given data
    def score(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        correct = 0
        total = 0
        # loop through each row to find the predicted label
        for i in range(0, len(Y)):
            vote, confidence = self.k_nearest_neighbor(X[i])
            # determine how many labels has been classified correctly
            if Y[i] == vote:
                correct += 1
            total += 1
        return correct / total
    
    # predict the label based off the given data
    def predict(self, x):
        X = np.array(x)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        result = []
        # loop through each row and provide the predicted label
        for val in X:
            vote, confidence = self.k_nearest_neighbor(val)
            result.append(vote)
        return result
    
    # predict the label based off given data and provide confidence
    def confidence(self, x):
        X = np.array(x)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        result = []
        # loop through each row and provide the predicted label with it's confidence
        for val in X:
            result.append(self.k_nearest_neighbor(val))
        return result
    
    # determine the most voted label with the confidence of that label
    def k_nearest_neighbor(self, x):
        X = np.array(x)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        distances = []
        # loop through each of the grouped labels
        for group in self.data_groups:
            # loop through each of the items within each of the labels
            for item in self.data_groups[group]:
                curr_item = []
                # find the distance between the target item and the provided data
                for i in range(0, len(item)):
                    curr = (float(X[i])-float(item[i]))**2
                    curr_item.append(curr)
                val = sqrt(sum(curr_item))
                # add the distance with the label into the list of distances array which will be used to find shortest distances
                distances.append([val, int(group)])
        # distances should be sorted in ascending order where it should only provide the the top "n_neighbors" (based on the "n_neighbors" provided)
        distances_sorted_k = sorted(distances, key=lambda x: x[0])[:self.n_neighbors]
        votes = []
        # ignore the distance as it should only provide the label which will be used to count how many times each label has occured
        for item in distances_sorted_k:
            votes.append(item[1])
        # this will get the most common label with the count of how many times it occured
        common_vote = Counter(votes).most_common()
        # calculate the confidence for that label
        confidence = common_vote[0][1] / self.n_neighbors
        return [int(common_vote[0][0]), confidence]