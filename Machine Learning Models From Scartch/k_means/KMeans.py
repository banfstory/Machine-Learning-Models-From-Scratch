from math import sqrt
import numpy as np
import warnings

class KMeans:
    def __init__(self, n_clusters=2, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    # this will fit the model with the given data
    def fit(self, x):
        X = np.array(x)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        self.centroids = {}
        # initialize centriods
        for i in range(0, self.n_clusters):
            self.centroids[i] = np.array(X[i])
        # keep looping until the cendriod stops moving or until the max iteraction count has been reached
        for i in range(0, self.max_iter):
            values_closest_centroids = {}
            # initialize this object which will be used to determine which values are closest to which centroids
            for key in self.centroids:
                values_closest_centroids[key] = []
            # loop through each row to determine the closest centroid
            for item in X:
                centroids_distance = {}
                # loop through the current centroids to get the distances
                for key, val in self.centroids.items():
                    curr_items = []
                    for index in range(0, len(item)):
                        curr = (item[index] - val[index])**2
                        curr_items.append(curr)
                    distance = sqrt(sum(curr_items))
                    centroids_distance[key] = distance
                # determine which centroid is closest for each row
                closest = min(centroids_distance, key=centroids_distance.get)
                values_closest_centroids[closest].append(item)
            for key, val in values_closest_centroids.items():
                values_closest_centroids[key] = np.array(val)
            new_centriods = {}
            # the new centroids will be the mean of the values grouped under that centroid
            for key, val in values_closest_centroids.items():
                # if there are no values grouped under a centroid then the new centroid should be the original centroid
                if(val.shape[0] == 0):
                    new_centriods[key] = np.array(self.centroids[key])
                new_centriods[key] = np.mean(val, axis=0)
            previous_centroids = self.centroids
            self.centroids = new_centriods
            # compare prev and curr centroid where if there are no changes to the centroids then it will break out of the loop
            compare_centroids = {}
            for key, val in new_centriods.items():
                compare_centroids[key] = sqrt(sum((val - previous_centroids[key]))**2)
            optimized = True
            for key in compare_centroids:
                if compare_centroids[key] != 0:
                    optimized = False
            if optimized:
                return self.centroids
        return self.centroids
            
    # predict the label based off the given data
    def predict(self, x):
        X = np.array(x)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        results = []
        # loop through each of the rows to determine the closest centroid
        for i in range(0, len(X)):
            target = X[i]
            distances = {}
            # loop through the centroids to get the distances
            for key, val in self.centroids.items():
                distance = sqrt(sum((val - target)**2))
                distances[key] = distance
            # determine which centroid is closest for each row
            results.append(min(distances, key=distances.get))
        return results