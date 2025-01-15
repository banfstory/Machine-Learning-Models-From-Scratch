import numpy as np
import math
import warnings

class NeuralNetwork():
    def __init__(self):
        self.model = []

    # this will add one layer to the neural network
    def add(self, layer):
        if type(layer).__name__ != Dense.__name__:
            return warnings.warn("The Dense class object must be used")
        self.model.append(layer)

    # this will determine the optmization method and the loss function that will be used
    def compile(self, optimizer='convex', loss='categorical_crossentropy', metrics=['accuracy']):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    # this will involve optimizing the neural networking which involves finding the best weights and biases that will minimize the loss
    def fit(self, x, y, iteraction=10000):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        # the label mapping ensures it is known which index of the final output produced by the neural network is referring to which label
        self.label_mapping = { }
        labels = np.unique(Y)
        # map the label to the appropriate index
        for i in range(0, len(labels)):
            self.label_mapping[labels[i]] = i
        # convert the features to the corresponding index which will simplify the calculation of the loss
        for i in range(0, len(Y)):
            Y[i] = self.label_mapping[Y[i]]
        self.input_count = X.shape[1]
        if len(self.model) == 0:
            return warnings.warn("Neural network has not added layers")
        # update the weights based off given input count for the first layer
        self.model[0].update_weights(self.input_count)
        # update the next layer's weights based off given output count from the previous layer which will be represented as the input
        for i in range(1, len(self.model)):
            self.model[i].update_weights(self.model[i - 1].neuron_count)
        self.optimize(X, Y, iteraction=iteraction)

    # determine the accuracy of the model based off given data
    def score(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        result = X
        correct = 0
        # convert the class label to the appropriate index, this will be used to compare with the output produced by the neural network
        for i in range(0, len(Y)):
            Y[i] = self.label_mapping[Y[i]]
        # run the neural network with the given inputs
        for layer in self.model:
            result = layer.output(result)
        # get the indexes with the highest probability for the inputs
        max_result = np.argmax(result, axis=1)
        # compare the predicted output with the actual output to get the total amount correct
        for i in range(0, len(max_result)):
            if Y[i] == max_result[i]:
                correct += 1
        return correct / len(Y)

    # predict the label based off the given data
    def predict(self, x):
        X = np.array(x)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        reverse_label_mapping = {}
        # the label mapping key and value needs to be reversed as we will need to print out the actual class label and not the index
        for (key, value) in enumerate(self.label_mapping):
            reverse_label_mapping[key] = value
        result = X
        for layer in self.model:
            result = layer.output(result)
        # get the indexes with the highest probability for the inputs
        max_result = np.argmax(result, axis=1)
        for i in range(0, len(max_result)):
            index = max_result[i]
            max_result[i] = reverse_label_mapping[index]
        return max_result
    
    def optimize(self, x, y, iteraction=10000):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 2:
            return warnings.warn("X array must be 2D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        optimal_loss = None
        count = 0
        # determine the optimization method in order provide the best weights and biases which minimizes the loss
        # if optimizer value provided does not exist then we are unable to perform the optimization for the neural network
        if self.optimizer == "convex":
            while count < iteraction:
                layer_details = []
                if optimal_loss != None:
                    # this involves making random changes to the weights and biases
                    for layer in self.model:
                        old_weights = np.array(layer.weights)
                        old_biases = np.array(layer.biases)
                        new_weights_add = np.random.randn(layer.weights.shape[0], layer.weights.shape[1])
                        new_biases_add = np.random.randn(layer.biases.shape[0])
                        layer_details.append({ "weights": old_weights, "biases": old_biases })
                        layer.weights += new_weights_add
                        layer.biases += new_biases_add
                result = X
                # loop through each layer within the neural network and pass the input which will either be the initial input or based off the previous output produced by the previous layer
                for layer in self.model:
                    result = layer.output(result)
                loss_function = Loss()
                # determine the loss function which will be used as a metric in order to optimize the neural network by finding the weights and biases which minimizes the loss
                # if the loss function value provided does not exist then we are unable to calculate the loss function therefore we also will not be able to perform the optimization
                if self.loss == "categorical_crossentropy":
                    loss = loss_function.categorical_crossentropy(result, Y)
                    # perform the action below if loss is an improvement to the previous loss
                    if optimal_loss == None or optimal_loss > loss:
                        optimal_loss = loss
                        # accuracy is optional to be shown as a metric
                        if "accuracy" in self.metrics:
                            max_result = np.argmax(result, axis=1)
                            total = 0
                            for i in range(0, len(Y)):
                                if Y[i] == max_result[i]:
                                    total += 1
                            accuracy = total / len(Y)
                        # print results to identify how each of the metrics are performing for each iteraction where the loss has improved
                        count_string = f"Iteraction: {count}"
                        accuracy_string = f"- Accuracy:{accuracy}" if "accuracy" in self.metrics else ""
                        loss_string = f"- Loss: {loss}"                  
                        print(f"{count_string} {loss_string} {accuracy_string}")
                    else:
                        # if changes to the weights and biases does not improve the loss then it will revert back to it's original weights and biases
                        for i in range(0, len(self.model)):
                            self.model[i].weights = layer_details[i]["weights"]
                            self.model[i].biases = layer_details[i]["biases"]
                else:
                    return warnings.warn("Invalid loss function provided")
                count += 1
        else:
            return warnings.warn("Invalid optimizer provided")

class Dense:
    # initialize the biases and number of neurons for the output which also includes the activation fucntion that will be ran against the output
    def __init__(self, neuron_count, activation):
        self.neuron_count = neuron_count
        self.activation = activation
        # bias will start at 0
        self.biases = np.zeros(neuron_count)
    
    # the weight should be the number of input times the number of neurons which will be randomized
    def update_weights(self, input_count):
        self.weights = np.random.rand(input_count, self.neuron_count)

    # this will take in an inputs which will be used to determine the output based on the bias, weights and activation function
    def output(self, input):
        # the inputs dot product with weights plus the bias
        output = np.dot(input, self.weights) + self.biases
        return self.activation(output)
    
class Activation():
    def relu(self, data):
        # if value is negative then convert to 0 otherwise keep as it's original value
        d = np.array(data)
        return np.maximum(0, d)

    def sigmoid(self, data):
        # the sigmoid formula is: 1 / (1 + e^(-x))
        d = np.array(data)
        d = d * -1
        d = np.exp(d)
        d = d + 1
        d = 1 / d
        return d

    def softmax(self, data):
        # the expoential must be caluated to ensure there are no negative numbers as the probability will need to be calulated which will be based off the total of the added values
        d = np.array(data)
        max = np.amax(d, axis = 1, keepdims=True)
        # subtracting by the maximum value will ensure that the exponent does not cause an overflow issue
        d = d - max
        exp = np.exp(d)
        sum = np.sum(exp, axis=1, keepdims=True)
        return exp / sum

class Loss:
    def categorical_crossentropy(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        x_label = []
        # to calculate the loss, we need to get the value of the input which is represented by the label
        for i in range(0, len(Y)):
            label = Y[i]
            x_label.append(X[i][label])
        x_label = np.array(x_label)
        # this will ensure that the minimum is never 0 (getting the log of 0 will cause the value to be infinite)
        x_label = np.clip(x_label, math.e**(-9), 1 - math.e**(-9))
        # changing the log of the value to a minus as the lower the probability the more negative the value which represents the loss, we would like to minimize that loss  
        loss = -np.log(x_label)
        return np.mean(loss)