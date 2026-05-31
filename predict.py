import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

# Load test data
test_data = pd.read_csv('dataset/test.csv')

# Predict
predictions = model.predict(test_data)

# Create submission
submission = pd.DataFrame({
    'target': predictions
})

submission.to_csv('submission.csv', index=False)

print('submission.csv created successfully - Untitled-1:20')