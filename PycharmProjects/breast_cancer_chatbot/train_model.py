import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('breast-cancer.csv')

# Preprocessing
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Malignant=1, Benign=0
X = data.drop(['id', 'diagnosis'], axis=1)  # Drop unnecessary columns
y = data['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'breast_cancer_model.pkl')
