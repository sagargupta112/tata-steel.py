import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
train_data = pd.read_csv('dataset/train.csv')

# Example target column
TARGET = 'target'

X = train_data.drop(TARGET, axis=1)
y = train_data[TARGET]

# Train test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Validation
preds = model.predict(X_valid)
print('Accuracy:  Untitled1:30 - train.py:30', accuracy_score(y_valid, preds))

# Save model
joblib.dump(model, 'model.pkl')

print('Model saved successfully  Untitled1:35 - train.py:35')