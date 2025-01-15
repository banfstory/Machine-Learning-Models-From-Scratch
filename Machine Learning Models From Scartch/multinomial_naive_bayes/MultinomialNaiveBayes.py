import numpy as np
import warnings

class MultinomialNaiveBayes:
    # this will fit the model with the given data
    def fit(self, x, y):
        self.frequency = {}
        self.label_details = {}
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        NUM_OF_STRINGS = 'num_of_strings'
        # loop through each label and initialize values such as total number of strings, count and probability
        for i in np.unique(Y):
            self.frequency[i] = {}
            self.label_details[i] = { 'num_of_strings': 0, 'count': 0, 'probability': 0 }
        # loop through each row and count the number of times each string occurs
        for i in range(0, len(Y)):
            label = Y[i]
            self.label_details[label]['count'] += 1
            split = X[i].split()
            # loop through each of the strings (separted by spaces)
            for s in split:
                # if string does not exist in frequency table than add it in
                if s not in self.frequency[label]:
                    # an additional count needs to be added to the occurence for all labels (this is to ensure that the probability for that string will never be 0 for the other labels as multiplying anything by 0 will still be 0 no matter how large the probability is)
                    self.frequency[label][s] = 2
                    self.label_details[label][NUM_OF_STRINGS] += 2
                    for l in self.frequency:
                        if l == self.frequency[label]:
                            continue
                        self.frequency[l][s] = 1
                        self.label_details[l][NUM_OF_STRINGS] += 1
                else:
                    # if the string already exist in the frequency table for that label then just increment it by 1
                    self.frequency[label][s] += 1
                    self.label_details[label][NUM_OF_STRINGS] += 1
        # calculate the probability of each label
        for label in self.label_details:
            self.label_details[label]['probability'] = self.label_details[label]['count'] / len(Y)

    # determine the accuracy of the model based off given data
    def score(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        correct = 0
        # loop through each row and determine which label it should be classified as
        for i in range(0, len(Y)):
            split = X[i].split()
            best_label = None
            best_score = None
            # loop through each label to find the best label based off it's score
            for label in self.label_details:
                # the first score should be initialized to the probability of a label occuring
                score = self.label_details[label]['probability']
                # the number of strings for that specific label will be used to calculate the probability
                num_of_strings = self.label_details[label]['num_of_strings']
                # loop through each string separated by spaces and calculate the probability for each string and add it to the current score
                for s in split:
                    # if string does not exist in the frequency table than it should be 1 divided by the total number of strings for that label for the probability
                    if s not in self.frequency[label]:
                        score *= (1 / num_of_strings)
                    else:
                        # if string exist then then find the probability of that specific string relative to the label and add it to the current score
                        score *= (self.frequency[label][s] / num_of_strings)
                # if a better label has been found based off the current score then update it as the new best label and score
                if best_label == None or score > best_score:
                    best_label = label
                    best_score = score
            # determine how many labels has been classified correctly
            if best_label == Y[i]:
                correct += 1
        return correct / len(Y)
        
    # predict the label based off the given data
    def predict(self, x):
        X = np.array(x)
        result = [None] * len(X)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        # loop through each row and determine which label it should be classified as
        for i in range(0, len(X)):
            split = X[i].split()
            best_label = None
            best_score = None
            # loop through each label to find the best label based off it's score
            for label in self.label_details:
                # the first score should be initialized to the probability of a label occuring
                score = self.label_details[label]['probability']
                # the number of strings for that specific label will be used to calculate the probability
                num_of_strings = self.label_details[label]['num_of_strings']
                # loop through each string separated by spaces and calculate the probability for each string and add it to the current score
                for s in split:
                    # if string does not exist in the frequency table than it should be 1 divided by the total number of strings for that label for the probability
                    if s not in self.frequency[label]:
                        score *= (1 / num_of_strings)
                    else:
                        # if string exist then then find the probability of that specific string relative to the label and add it to the current score
                        score *= (self.frequency[label][s] / num_of_strings)
                # if a better label has been found based off the current score then update it as the new best label and score
                if best_label == None or score > best_score:
                    best_label = label
                    best_score = score
            # record this row with the predicted label
            result[i] = int(best_label)
        return result