from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
    float(request.form['Pregnancies']),
    float(request.form['Glucose']),
    float(request.form['BloodPressure']),
    float(request.form['SkinThickness']),
    float(request.form['Insulin']),
    float(request.form['BMI']),
    float(request.form['DiabetesPedigreeFunction']),
    float(request.form['Age']),
    float(request.form['HbA1c_level']),
    float(request.form['blood_glucose_level']),
    float(request.form['hypertension']),
    float(request.form['heart_disease']),
    float(request.form['smoking_history_former'])  # ✅ Correct 13th feature
]


    # Optional: remove the duplicate BMI if your model was trained with only one BMI feature
    input_data = input_data[:-1]  # removes the second BMI

    input_array = np.asarray(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    result = 'The person is Diabetic' if prediction == 1 else 'The person is Not Diabetic'
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
