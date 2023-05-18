import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load the dataset from a CSV file
data = pd.read_csv("data/fraud_email_.csv")

# Drop rows with missing values
data = data.dropna()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Class'], test_size=0.2, random_state=42)

# Create a feature extractor
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train the model
model = svm.SVC()
model.fit(X_train_features, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_features)

# Print the predictions


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print(model.predict(vectorizer.transform(["""Subject: Urgent Request - Immediate Action Required

Dear Customer,

We have noticed some unusual activities in your account. To ensure the security of your account, we request you to verify your information by clicking on the following link:

www.youtube.com

Failure to verify your account within 24 hours may result in a temporary suspension of your account. We value your security and appreciate your immediate attention to this matter.

Thank you,
Customer Support
"""])))