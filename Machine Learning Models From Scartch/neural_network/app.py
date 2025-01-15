import os
import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork, Dense, Activation
    
# this is the data that will be processed into the neural network
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'breast-cancer-wisconsin.data')) # the source of the data came from this link: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
x = df.drop(['class'],axis=1)
x = x.astype(float)
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
neuron_count = int(x_train.shape[1] * (3/4) + len(y_test.unique()))
output_count = len(y_test.unique())

# this involves setting up the neural network and feeding the data into the neural network
activation = Activation()
model = NeuralNetwork()
model.add(Dense(neuron_count, activation.relu))
model.add(Dense(neuron_count, activation.relu))
model.add(Dense(output_count, activation.softmax))
model.compile(optimizer='convex', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, iteraction=20000)
print(f"Accuracy: {model.score(x_test, y_test)}")
print(f"Prediction: {model.predict(x_test)}")