from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from MultinomialNaiveBayes import MultinomialNaiveBayes

# this is the data that will be processed into the model
df = pd.read_csv(os.path.join(os.path.dirname(__file__),'spam.csv'))
label_enc = LabelEncoder()
df['Category'] = label_enc.fit_transform(df['Category'])
x = df['Message']
y = df['Category']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.20)

# this involves fitting the model with the data
model = MultinomialNaiveBayes()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predict = model.predict(x_test)
print(f"Accuracy: {score}")
print(f"Prediction: {predict}")