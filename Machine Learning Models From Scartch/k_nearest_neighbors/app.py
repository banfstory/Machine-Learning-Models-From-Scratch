import pandas as pd
from sklearn import model_selection
import os
from KNearestNeighbors import KNearestNeighbors

# this is the data that will be processed into the model
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'breast-cancer-wisconsin.data')) # the source of the data came from this link: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
x = df.drop(['class'],axis=1)
y = df['class']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

# this involves fitting the model with the data
model = KNearestNeighbors(n_neighbors=3)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predict = model.predict(x_test)
confidence = model.confidence(x_test)
print(f"Accuracy: {score}")
print(f"Prediction: {predict}")
print(f"Confidence: {confidence}")