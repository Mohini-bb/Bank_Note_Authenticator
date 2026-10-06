import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())

# Train test split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model training
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score*100:.2f}%")

# Save model
pickle.dump(classifier, open('classifier.pkl', 'wb'))
print("SUCCESS! model saved as classifier.pkl")