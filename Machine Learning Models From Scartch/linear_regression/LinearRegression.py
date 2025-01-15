import numpy as np
import warnings

class LinearRegression:
    # this will fit the model with the given data
    def fit(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        # find the mean for both the x and y axis which will represent one of the points of the linear regression line
        self.x_mean = np.array(X).mean()
        self.y_mean = np.array(Y).mean()
        # find the slope
        self.m = self.slope(X,Y)
        # find the y-intercept
        self.b = self.y_intercept()

    # determine the accuracy of the model based off given data
    def score(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        if len(Y.shape) != 1:
            return warnings.warn("Y array must be 1D array")
        se_line = self.square_error_line(X, Y)
        se_mean = self.square_error_mean(Y)
        # uses r-squared to determine how well the data fits the given regression line
        return 1 - (se_line / se_mean)
    
    # predict the label based off the given data
    def predict(self, x):
        X = np.array(x)
        if len(X.shape) != 1:
            return warnings.warn("X array must be 1D array")
        results = []
        for val in X:
            results.append(float(self.y_regression(val)))
        return results

    # this is the formula to find the slope of the linear regression line
    def slope(self, x,y):
        return ((self.x_mean*self.y_mean)-((x*y).mean()))/((self.x_mean**2)-((x**2).mean()))

    # this is to find the y-intercept of the linear regression line where y mean and x mean are one point on the line
    def y_intercept(self):
        return self.y_mean-self.m*self.x_mean

    # this will provide the prediction of the value of y
    def y_regression(self,x):
        return self.m*x+self.b
    
    # this represents the square error of linear regression line
    def square_error_line(self, x, y):
        return ((y - self.y_regression(x)) ** 2).sum()

    # this represents the square error of the mean line
    def square_error_mean(self, y):
        return ((y - self.y_mean) ** 2).sum()