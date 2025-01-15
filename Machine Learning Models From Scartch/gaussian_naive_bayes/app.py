from sklearn import model_selection
import pandas as pd
import os
from GaussianNaiveBayes import GaussianNaiveBayes

# this is the data that will be processed into the model
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'breast-cancer-wisconsin.data')) # the source of the data came from this link: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
df.loc[(df['class'] == 2), 'class'] = 0
df.loc[(df['class'] == 4), 'class'] = 1
df = df.astype(int)
x = df.drop(['class'],axis=1)
y = df['class']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

# this involves fitting the model with the data
model = GaussianNaiveBayes()
model.fit(x_train, y_train)
print(f"Accuracy: {model.score(x_test, y_test)}")
print(f"Prediction: {model.predict(x_test)}")