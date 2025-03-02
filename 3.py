import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('data.csv')
print("the values of the data is\n " ,data )

# Encode categorical features
X = data.iloc[:, :-1].apply(LabelEncoder().fit_transform)
print("\nThe First 5 values of train data is\n",X.head()) 

y = LabelEncoder().fit_transform(data.iloc[:, -1])
print("\nThe output of the train data is\n",y)
# Drop zero variance columns
X = X.loc[:, X.var() > 0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

# Train and evaluate model with var_smoothing
classifier = GaussianNB(var_smoothing=1e-9)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Display results
print("Actual:", y_test)
print("Predicted:", predictions)
print("Accuracy:", accuracy_score(predictions, y_test))
