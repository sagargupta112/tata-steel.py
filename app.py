from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)

        prediction = model.predict(final_input)

        return render_template(
            'index.html',
            prediction_text=f'Prediction Result: {prediction[0]}'
        )
 except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == '__main__':
    app.run(debug=True)