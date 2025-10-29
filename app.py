"""
Flask Web Application for Career Path Analysis
Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import json
import traceback

app = Flask(__name__)

# Global variables
model = None
scaler = None
feature_names = None
dataset_info = {}
df_original = None

def load_and_train_model():
    """Load dataset and train model"""
    global model, scaler, feature_names, dataset_info, df_original
    
    try:
        # Load dataset
        df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
        df_original = df.copy()
        
        # Store dataset info
        attrition_yes = (df['Attrition'] == 'Yes').sum()
        attrition_no = (df['Attrition'] == 'No').sum()
        
        dataset_info = {
            'total_employees': int(len(df)),
            'attrition_count': int(attrition_yes),
            'retention_count': int(attrition_no),
            'attrition_rate': f"{(attrition_yes / len(df) * 100):.2f}",
            'avg_age': f"{df['Age'].mean():.1f}",
            'avg_income': f"{df['MonthlyIncome'].mean():.2f}",
            'departments': df['Department'].unique().tolist() if 'Department' in df.columns else [],
            'job_roles': df['JobRole'].unique().tolist() if 'JobRole' in df.columns else []
        }
        
        # Preprocess
        df_processed = df.copy()
        
        # Drop unnecessary columns
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)
        
        # Encode target
        df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            df_processed[col] = le.fit_transform(df_processed[col])
        
        # Separate features and target
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("✓ Model trained successfully!")
        print(f"✓ Features: {len(feature_names)}")
        print(f"✓ Dataset info loaded: {dataset_info}")
        return True
        
    except FileNotFoundError:
        print("❌ Dataset not found. Please download it from Kaggle.")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', dataset_info=dataset_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict attrition for a single employee"""
    try:
        data = request.json
        
        # Create feature vector in correct order
        features = []
        for feature in feature_names:
            features.append(float(data.get(feature, 0)))
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'attrition_probability': f"{probability[1] * 100:.2f}%",
            'retention_probability': f"{probability[0] * 100:.2f}%",
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict attrition for multiple employees from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Preprocess the uploaded data similar to training
        df_processed = df.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = le.fit_transform(df_processed[col])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df_processed.columns)
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # Select and order features
        X = df_processed[feature_names]
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Add results to dataframe
        df['Attrition_Prediction'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
        df['Attrition_Probability'] = [f"{p*100:.2f}%" for p in probabilities]
        
        # Save results
        output_path = 'predictions.csv'
        df.to_csv(output_path, index=False)
        
        summary = {
            'total_employees': len(df),
            'high_risk_count': int((predictions == 1).sum()),
            'low_risk_count': int((predictions == 0).sum()),
            'avg_risk_probability': f"{probabilities.mean() * 100:.2f}%"
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'download_url': '/download_predictions'
        })
        
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/download_predictions')
def download_predictions():
    """Download prediction results"""
    return send_file('predictions.csv', as_attachment=True)

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html', dataset_info=dataset_info)

@app.route('/api/feature_importance')
def feature_importance():
    """Get feature importance data"""
    try:
        if model is None:
            return jsonify({'error': 'Model not trained'}), 400
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        data = {
            'features': [feature_names[i] for i in indices],
            'importances': [float(importances[i]) for i in indices]
        }
        
        return jsonify(data)
    except Exception as e:
        print(f"Feature importance error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset_stats')
def dataset_stats():
    """Get dataset statistics"""
    try:
        if not dataset_info:
            return jsonify({'error': 'Dataset not loaded'}), 400
        
        # Add department data if available
        stats = dataset_info.copy()
        
        if df_original is not None and 'Department' in df_original.columns:
            dept_counts = df_original['Department'].value_counts().to_dict()
            stats['department_counts'] = {str(k): int(v) for k, v in dept_counts.items()}
        
        return jsonify(stats)
    except Exception as e:
        print(f"Dataset stats error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CAREER PATH ANALYSIS - WEB APPLICATION")
    print("="*80)
    print("\nInitializing...")
    
    if load_and_train_model():
        print("\n✓ Server starting...")
        print("✓ Access the application at: http://127.0.0.1:5000")
        print("✓ API endpoints available:")
        print("  - /api/feature_importance")
        print("  - /api/dataset_stats")
        print("\nPress CTRL+C to stop the server\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to start application. Please check the dataset.")
        print("Download from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")