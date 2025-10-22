from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'simple_key'

# Load model and encoders
try:
    model = joblib.load('models/decision_tree_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    le_sex = encoders['sex']
    le_embarked = encoders['embarked']
except FileNotFoundError:
    model = None

class DataPreprocessor:
    @staticmethod
    def validate(data):
        """Validate input fields and ranges"""
        try:
            pclass = int(data['Pclass'])
            if pclass not in [1, 2, 3]:
                raise ValueError("Pclass must be 1, 2, or 3")
            
            sex = data['Sex'].lower()
            if sex not in ['male', 'female']:
                raise ValueError("Sex must be male or female")
            
            age = float(data['Age'])
            if not 0 <= age <= 100:
                raise ValueError("Age must be 0-100")
            
            sibsp = int(data['SibSp'])
            if not 0 <= sibsp <= 8:
                raise ValueError("SibSp must be 0-8")
            
            parch = int(data['Parch'])
            if not 0 <= parch <= 6:
                raise ValueError("Parch must be 0-6")
            
            fare = float(data['Fare'])
            if not 0 <= fare <= 600:
                raise ValueError("Fare must be 0-600")
            
            embarked = data['Embarked'].upper()
            if embarked not in ['C', 'Q', 'S']:
                raise ValueError("Embarked must be C, Q, or S")
            
            return True
        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {str(e)}")
            return False

    @staticmethod
    def transform(data):
        """Transform to DataFrame with feature names (fixes warning)"""
        sex_encoded = le_sex.transform([data['Sex'].lower()])[0]
        embarked_encoded = le_embarked.transform([data['Embarked'].upper()])[0]
        
        # Create DataFrame with proper column names
        features_df = pd.DataFrame([[
            int(data['Pclass']), 
            sex_encoded, 
            float(data['Age']),
            int(data['SibSp']), 
            int(data['Parch']), 
            float(data['Fare']),
            embarked_encoded
        ]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
        
        return features_df

class SurvivalModel:
    @staticmethod
    def predict(input_dict):
        """Predict survival and provide explanation"""
        if model is None:
            raise ValueError("Model not loaded")
        
        features_df = DataPreprocessor.transform(input_dict)
        pred = model.predict(features_df)[0]
        result = "✓ Survived" if pred == 1 else "✗ Did not survive"
        explanation = f"Based on: Class {input_dict['Pclass']}, Gender {input_dict['Sex']}, Age {input_dict['Age']}"
        
        return result, explanation

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        if DataPreprocessor.validate(data):
            try:
                result, explanation = SurvivalModel.predict(data)
                return render_template('result.html', result=result, explanation=explanation)
            except Exception as e:
                flash(f"Prediction failed: {str(e)}")
                return redirect(url_for('predict'))
        return redirect(url_for('predict'))
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)