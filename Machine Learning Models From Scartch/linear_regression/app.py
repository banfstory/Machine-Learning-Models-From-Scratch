from matplotlib import pyplot as plt
from sklearn import model_selection
import pandas as pd
from LinearRegression import LinearRegression

# this is the data that will be processed into the model
df = pd.read_csv('customer_purchases.csv') # the source of the data came from this link: https://www.kaggle.com/datasets/hanaksoy/customer-purchasing-behaviors
x = df['age']
y = df['annual_income']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

# this involves fitting the model with the data
model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predict = model.predict(x_test)
print(f"Accuracy: {score}")
print(f"Prediction: {predict}")

# plot the regression line
plt.title("Annual Income by Age")
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.plot(x_test, predict, color="black", label="Regression line")
plt.scatter(x_test, y_test, label="Test data")
plt.legend()
plt.show()
