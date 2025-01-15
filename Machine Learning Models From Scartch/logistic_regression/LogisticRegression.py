import numpy as np
from math import sqrt, e, log
import warnings

class LogisticRegression:
    # this will fit the model with the given data which involves transforming the linear line into the sigmoid curve (regression logistic model)
    # the linear line (where the y axis represents the "log odds") will be used to calculate the probability through the sigmoid curve
    def fit(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        unique = np.unique(Y)
        # the unique values must be 0 or 1
        if unique.shape[0] != 2:
            return warnings.warn("Y must have 2 unique values")
        if unique.min() != 0 and unique.max() != 1:
            return warnings.warn("Y must have unique values of 0 and 1")
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        # the min and max will be used to visually graph the sigmoid curve
        self.x_min = np.min(X)
        self.x_max = np.max(X)
        # find the mean, standard deviation, slope and y-intercept which will be used to calculate the best regression logistic model to use
        self.mean(X, Y)
        self.std(X)
        self.slope()
        self.y_intercept()
        # optimize the logistics regression model
        self.optimize(X, Y)

    # determine the accuracy of the model based off given data
    def score(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        accuracy = 0
        for i in range(0, len(Y)):
            probability = self.probability(X[i])
            if probability >= 0.50:
                if Y[i] == 1:
                    accuracy += 1
            elif Y[i] == 0:
                accuracy += 1
        return accuracy / len(Y)

    # predict the label based off the given data
    def predict(self, x):
        result = []
        probability_list = []
        X = np.array(x)
        for val in X:
            probability = self.probability(val)
            probability_list.append(float(probability))
            if probability >= 0.50:
                result.append(1)
            else:
                result.append(0)
        return result, probability_list
    
    # the mean will represent a point on the linear line and the sigmoid curve which will remain the same during the optimization process (this point will represent the probability of 0.5)
    def mean(self, x, y):
        true_total = 0
        true_count = 0
        false_total = 0
        false_count = 0
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        # keep track of the count and total for the two labels (0 or 1)
        for i in range(0, len(Y)):
            if Y[i] == 1:
                true_total += X[i]
                true_count += 1
            else:
                false_total += X[i]
                false_count += 1
        # this will represent the two means
        true_mean = true_total / true_count
        false_mean = false_total / false_count
        # the true mean will represent the mid point of the two means for the linear line and sigmoid curve
        average_mean = (true_mean + false_mean) / 2
        self.mean = average_mean

    # represents standard deviation (which is used to get the slope) of the linear line
    def std(self, x):
        X = np.array(x)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        total = 0
        # loop through each row to calculate the standard deviation
        for val in X:
            total += (val - self.mean) ** 2
        self.std = sqrt(total / len(X))

    # the slope for the linear line (this will be constantly changed throughout the optmization process)
    def slope(self):
        # full formula - slope = (1 - 0) / ((self.mean + self.std) - self.mean)
        self.slope = 1 / self.std
    
    # the y-intercept for the linear line (this will be constantly changed throughout the optmization process)
    def y_intercept(self):
        # full formula - y_intercept = 0 - (self.slope * self.mean)
        self.y_intercept = -(self.slope * self.mean)

    # find the probability which will be used to calculate the accuracy and predictions
    def probability(self, x):
        # find the log odds which is represented by the linear line
        log_odds = (self.slope * x) + self.y_intercept
        # find the probability which is represented by the sigmoid curve once the log odds is found
        return (e**log_odds) / (1+(e**log_odds))

    # find the probability which will be used to optimize logistic regression model
    def probability_precalculate(self, x, slope, y_intercept):
        # find the log odds which is represented by the linear line
        log_odds = (slope * x) + y_intercept
        # find the probability which is represented by the sigmoid curve once the log odds is found
        return (e**log_odds) / (1+(e**log_odds))   

    # optimize the logistic regression model which involves finding the best linear line which will be transformed into the sigmoid curve to determine the probability
    def optimize(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        # index 0 represents whether to increase or reduce the slope
        # index 1 represents whether to look at the positive or negative slope
        direction = [[0,1],[0,-1],[1,1],[1,-1]]
        slope = self.slope
        y_intecept = self.y_intercept
        best_score = float('-inf')
        best_slope = self.slope
        best_y_intercept = self.y_intercept
        # find the best slope and y-intercept that fit the best
        for i in range(1, 11):
            for d in direction:
                scale = ((0.1 * i) + d[0]) * d[1]
                curr_slope = slope * scale
                curr_y_intercept = y_intecept * scale
                curr_score = 0
                # loop through each row and find the probability and add the log probability to the current score
                for j in range(0, len(Y)):
                    curr_x = X[j]
                    curr_y = Y[j]
                    probability = self.probability_precalculate(curr_x, curr_slope, curr_y_intercept)
                    # the calculation is different based on whether it is 1 or 0
                    if curr_y == 1:
                        curr_score += log(probability)
                    elif curr_y == 0:
                        curr_score += log(1 - probability)
                # check if the current score is better, if it is better than the current score becomes the new best score along with the new best slope and y-intercept
                if curr_score > best_score:
                    best_score = curr_score
                    best_slope = curr_slope
                    best_y_intercept = curr_y_intercept
        # the best slope and best y-intercept becomes the new slope and y-intercept
        self.slope = best_slope
        self.y_intercept = best_y_intercept
    
    # this involves graphing the sigmoid curve
    def plot_graph(self, plot_object):
        x_cords = range(self.x_min, self.x_max)
        y_cords = [self.probability(x) for x in x_cords]
        plot_object.plot(x_cords, y_cords, color="black")