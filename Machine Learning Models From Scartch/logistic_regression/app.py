import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
from LogisticRegression import LogisticRegression
import numpy as np

# plot the x and y
def plot_graph(x, y):
    X = np.array(x)
    Y = np.array(y)
    group_true_x = []
    group_true_y = []
    group_false_x = []
    group_false_y = []
    # loop through each row and group into either true or false (the y-axis will represent either 0 or 1)
    for i in range(0, len(Y)):
        if Y[i] == 1:
            group_true_x.append(X[i])
            group_true_y.append(1)
        else:
            group_false_x.append(X[i])
            group_false_y.append(0)
    #plot the points
    plt.scatter(group_true_x, group_true_y, color='b')
    plt.scatter(group_false_x, group_false_y, color='r')

# find the probability of y based on given x and then plot it
def plot_graph_probability(x, y):
    X = np.array(x)
    Y = np.array(y)
    group_true_x = []
    group_true_y = []
    group_false_x = []
    group_false_y = []
    # loop through each row and group into either true or false (the y-axis will represent the probability)
    for i in range(0, len(Y)):
        if Y[i] >= 0.5:
            group_true_x.append(X[i])
            group_true_y.append(Y[i])
        else:
            group_false_x.append(X[i])
            group_false_y.append(Y[i])
    # plot the points
    plt.scatter(group_true_x, group_true_y, color='b', marker='x', label="Insurance")
    plt.scatter(group_false_x, group_false_y, color='r', marker='x', label="No insurance")

# this is the data that will be processed into the model
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'insurance.csv'))
x = df['age']
y = df['bought_insurance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# this involves fitting the model with the data
model = LogisticRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predict, probability = model.predict(x_test)
print(f"Accuracy: {score}")
print(f"Prediction: {predict}")

# plot the logistic regression curve
model.plot_graph(plt)
plt.title("People who bought insurance by Age")
plt.xlabel("Age")
plt.ylabel("Probability")
plot_graph(x_train, y_train)
plot_graph_probability(x_test, probability)
plt.legend()
plt.show()