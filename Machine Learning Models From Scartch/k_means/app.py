from matplotlib import pyplot as plt
import numpy as np
from KMeans import KMeans

# this is the data that will be processed into the model
x_train = np.array([
    [1, 2, 3],
    [4, 3, 4],
    [5, 8, 5],
    [8, 8, 7],
    [2, 1, 6],
    [9, 11, 4],
    [11, 11, 2],
    [3, 5, 1]
])
x_test = np.array([
    [10,10,5], 
    [2,2,2], 
    [3,3,3],
    [7, 7, 6]
])

# this involves fitting the model with the data
clf = KMeans(n_clusters=2)
centroids = clf.fit(x_train)
predict_train = clf.predict(x_train)
predict_test = clf.predict(x_test)

def plot_graph(x, y):
    item_group = { }
    unique = np.unique(y)
    for label in unique:
        item_group[label] = []
    for i in range(0, len(x)):
        item = x[i]
        label = y[i]
        item_group[label].append(item)
    return item_group

# plot the 3D graph
colors = { 0:'r',1:'g' }
plt.axes(projection='3d')
# represents the centroids
for key, value in centroids.items():
    plt.scatter(value[0], value[1], value[2], color= colors[key], linewidths=10, marker="x", label=f"Centroid {key}")

# represents the training data
training_data = plot_graph(x_train, predict_train)
for key, value in training_data.items():
    item = np.column_stack(value)
    plt.scatter(item[0], item[1], item[2], color=colors[key], linewidths=5, marker='o', label=f"Data (Centroid {key})")
    
"""
# represents the test data
testing_data = plot_graph(x_test, predict_test)
for key, value in testing_data.items():
    item = np.column_stack(value)
    plt.scatter(item[0], item[1], item[2], color=colors[key], linewidths=5, marker='o', label=f"Test Data (Centroid {key})")
"""
plt.legend(loc='upper left')
plt.show()