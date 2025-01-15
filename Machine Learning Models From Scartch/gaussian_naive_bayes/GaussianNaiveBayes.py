import numpy as np
import math
import warnings

class GaussianNaiveBayes:
    def __init__(self):
        # this will be the model to make the prediction
        self.model = {}

    # this will fit the model with the given data
    def fit(self, x, y):
        self.model = {}
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 2:
            print("The shape should be 2 dimensional")
            return
        row_count = X.shape[0]
        column_count = X.shape[1]
        self.columns = X.shape[1]
        label_details = {}
        # initialize each feature (column) of each label into an array of objects
        for label in np.unique(Y):
            label_details[label] = {'columns': [], 'row_count': 0, 'probability': 0}
            for i in range(0, column_count):
                label_details[label]['columns'].append({ 'mean': 0, 'sd': 0, 'total': 0, 'items': []})
            self.model[label] = { 'columns': [None] * column_count, 'row_count': 0 }
        # finding total for each feature (column) of each label which will be used to calculate the mean
        for i in range(0, row_count):
            label = Y[i]
            target_X = X[i]
            target_label_details = label_details[label]
            target_label_details['row_count'] += 1
            for j in range(0, len(target_X)):
                target_column = target_label_details['columns'][j]
                target_column['total'] += target_X[j]
                target_column['items'].append(target_X[j])
        # finding mean for each feature (column) of each label which will be used to calculate the standard deviation
        for i in label_details:
            target_label_details = label_details[i]
            target_row_count = target_label_details['row_count']
            for target_row in target_label_details['columns']:
                target_row['mean'] = target_row['total'] / target_row_count
        # finding standard deviation for each feature (column) of each label
        for i in label_details:
            target_label_details = label_details[i]
            target_row_count = target_label_details['row_count']
            for target_row in target_label_details['columns']:
                mean = target_row['mean']
                total = 0
                for val in target_row['items']:
                    total += ((val - mean) ** 2)
                target_row['sd'] = math.sqrt(total / target_row_count)
        # create the model
        for i in label_details:
            target_label_details = label_details[i]
            target_columns = target_label_details['columns']
            # this represents the total rows for this label
            self.model[i]['row_count'] = target_label_details['row_count']
            # this will find the probability of this label being selected
            self.model[i]['probability'] = target_label_details['row_count'] / row_count
            # this will add the model for each of the features to determine the likelihood of each of the features
            for col in range(0, len(target_columns)):
                target_column = target_columns[col]
                model = GaussianDistribution(target_column['mean'], target_column['sd'])
                self.model[i]['columns'][col] = model

    # determine the accuracy of the model based off given data
    def score(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        correct = 0
        if X.shape[1] != self.columns:
            print(f'X must have ${self.columns} columns')
            return
        # the label with the highest score will be the chosen label (this involves adding up all the likelihood of each of the features to determine the score)
        for i in range(0, len(Y)):
            best_label = None
            best_score = None
            target_X = X[i]
            # loop through each of the labels and find the score (likelihood added together) where the highest score will be considered the best label
            for j in self.model:
                target_label = self.model[j]
                probability = math.log(target_label['probability'])
                for k in range (0, len(target_label['columns'])):  
                    probability += math.log(target_label['columns'][k].gaussian_distribution(target_X[k]))
                if best_label == None or best_score < probability:
                    best_label = j
                    best_score = probability
            # determine how many of those labels were classified correctly
            if best_label == Y[i]:
                correct += 1
        return correct / len(Y)

    # predict the label based off the given data
    def predict(self, x):
        X = np.array(x)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        result = [None] * len(X)
        if X.shape[1] != self.columns:
            print(f'X must have ${self.columns} columns')
            return
        # the label with the highest score will be the chosen label (this involves adding up all the likelihood of each of the features to determine the score)
        for i in range(0, len(X)):
            best_label = None
            best_score = None
            target_X = X[i]
            # loop through each of the labels and find the score (likelihood added together) where the highest score will be considered the best label
            for j in self.model:
                target_label = self.model[j]
                probability = math.log(target_label['probability'])
                for k in range (0, len(target_label['columns'])):     
                    probability += math.log(target_label['columns'][k].gaussian_distribution(target_X[j]))
                if best_label == None or best_score < probability:
                    best_label = j
                    best_score = probability
            result[i] = int(best_label)
        return result
    
class GaussianDistribution:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    # gaussian distribution formula to determine the likelihood
    def gaussian_distribution(self, x):
        return (1/(self.sd*math.sqrt(2*math.pi)))*math.e**((-1/2)*(((x-self.mean)/self.sd))**2)