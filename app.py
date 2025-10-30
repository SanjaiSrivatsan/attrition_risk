"""
Flask Web Application for Career Path Analysis
Complete version with integrated training option
Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import traceback
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
model = None
scaler = None
feature_names = None
model_metadata = None
dataset_info = {}
df_original = None

def train_and_save_model(data_path='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """Train all models, select best, and save"""
    print("\n" + "="*80)
    print("TRAINING NEW MODEL")
    print("="*80)
    
    try:
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"✓ Loaded dataset: {df.shape}")
        
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
        label_encoders = {}
        
        for col in categorical_cols:
            le_col = LabelEncoder()
            df_processed[col] = le_col.fit_transform(df_processed[col])
            label_encoders[col] = le_col
        
        # Separate features and target
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        feature_names_list = X.columns.tolist()
        print(f"✓ Features: {len(feature_names_list)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        print("✓ Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler_obj = StandardScaler()
        X_train_scaled = scaler_obj.fit_transform(X_train_res)
        X_test_scaled = scaler_obj.transform(X_test)
        
        # Train all models
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        best_score = 0
        best_model = None
        best_model_name = None
        
        for name, model_obj in models.items():
            print(f"\n→ Training {name}...")
            
            # Train
            model_obj.fit(X_train_scaled, y_train_res)
            
            # Predict
            y_pred = model_obj.predict(X_test_scaled)
            y_pred_proba = model_obj.predict_proba(X_test_scaled)[:, 1] if hasattr(model_obj, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'model': model_obj
            }
            
            # Composite score for selection
            composite_score = f1 * 0.4 + auc * 0.4 + accuracy * 0.2
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = model_obj
                best_model_name = name
        
        # Print best model
        print("\n" + "="*80)
        print("BEST MODEL SELECTED")
        print("="*80)
        print(f"\n✓ WINNER: {best_model_name}")
        print(f"  - Accuracy:  {results[best_model_name]['accuracy']:.4f}")
        print(f"  - Precision: {results[best_model_name]['precision']:.4f}")
        print(f"  - Recall:    {results[best_model_name]['recall']:.4f}")
        print(f"  - F1-Score:  {results[best_model_name]['f1_score']:.4f}")
        print(f"  - ROC-AUC:   {results[best_model_name]['auc']:.4f}")
        
        # Create models directory
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save best model
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler_obj, f)
        
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names_list, f)
        
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'accuracy': results[best_model_name]['accuracy'],
            'precision': results[best_model_name]['precision'],
            'recall': results[best_model_name]['recall'],
            'f1_score': results[best_model_name]['f1_score'],
            'roc_auc': results[best_model_name]['auc'],
            'n_features': len(feature_names_list),
            'training_samples': len(X_train_scaled),
            'test_samples': len(X_test)
        }
        
        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\n✓ All model files saved to models/ directory")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        traceback.print_exc()
        return False

def load_trained_model():
    """Load the pre-trained model and preprocessing objects"""
    global model, scaler, feature_names, model_metadata, dataset_info, df_original
    
    try:
        print("\n" + "="*80)
        print("LOADING MODEL")
        print("="*80)
        
        # Check if model files exist
        model_files = [
            'models/best_model.pkl',
            'models/scaler.pkl',
            'models/feature_names.pkl',
            'models/model_metadata.pkl'
        ]
        
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            print("\n⚠️  Model files not found. Training new model...")
            
            # Check if dataset exists
            if not os.path.exists('WA_Fn-UseC_-HR-Employee-Attrition.csv'):
                print("\n❌ Dataset not found!")
                print("Please download: WA_Fn-UseC_-HR-Employee-Attrition.csv")
                return False
            
            # Train new model
            if not train_and_save_model():
                return False
            
            print("\n✓ Model trained successfully! Loading it now...")
        
        # Load model
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Loaded model")
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Loaded scaler")
        
        # Load feature names
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ Loaded {len(feature_names)} features")
        
        # Load metadata
        with open('models/model_metadata.pkl', 'rb') as f:
            model_metadata = pickle.load(f)
        print(f"✓ Model: {model_metadata['model_name']}")
        print(f"  Accuracy: {model_metadata['accuracy']:.4f}")
        print(f"  F1-Score: {model_metadata['f1_score']:.4f}")
        
        # Load dataset for statistics
        df_original = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
        
        # Calculate dataset info
        attrition_yes = (df_original['Attrition'] == 'Yes').sum()
        attrition_no = (df_original['Attrition'] == 'No').sum()
        
        dataset_info = {
            'total_employees': int(len(df_original)),
            'attrition_count': int(attrition_yes),
            'retention_count': int(attrition_no),
            'attrition_rate': f"{(attrition_yes / len(df_original) * 100):.2f}",
            'avg_age': f"{df_original['Age'].mean():.1f}",
            'avg_income': f"{df_original['MonthlyIncome'].mean():.2f}",
            'departments': df_original['Department'].unique().tolist() if 'Department' in df_original.columns else [],
            'job_roles': df_original['JobRole'].unique().tolist() if 'JobRole' in df_original.columns else [],
            'model_name': model_metadata['model_name'],
            'model_accuracy': f"{model_metadata['accuracy']*100:.2f}",
            'model_f1_score': f"{model_metadata['f1_score']*100:.2f}",
            'model_precision': f"{model_metadata['precision']*100:.2f}",
            'model_recall': f"{model_metadata['recall']*100:.2f}",
            'model_roc_auc': f"{model_metadata['roc_auc']*100:.2f}"
        }
        
        print("✓ Dataset loaded")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
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
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low',
            'model_used': model_metadata['model_name']
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
        df['Model_Used'] = model_metadata['model_name']
        
        # Save results
        output_path = 'predictions.csv'
        df.to_csv(output_path, index=False)
        
        summary = {
            'total_employees': len(df),
            'high_risk_count': int((predictions == 1).sum()),
            'low_risk_count': int((predictions == 0).sum()),
            'avg_risk_probability': f"{probabilities.mean() * 100:.2f}%",
            'model_used': model_metadata['model_name']
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
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            data = {
                'features': [feature_names[i] for i in indices],
                'importances': [float(importances[i]) for i in indices]
            }
        else:
            # For models without feature importance (e.g., Logistic Regression)
            if hasattr(model, 'coef_'):
                coef = np.abs(model.coef_[0])
                indices = np.argsort(coef)[::-1][:15]
                data = {
                    'features': [feature_names[i] for i in indices],
                    'importances': [float(coef[i]) for i in indices]
                }
            else:
                return jsonify({'error': 'Model does not support feature importance'}), 400
        
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

@app.route('/api/model_info')
def model_info():
    """Get current model information"""
    try:
        if model_metadata is None:
            return jsonify({'error': 'Model not loaded'}), 400
        
        info = {
            'model_name': model_metadata['model_name'],
            'accuracy': f"{model_metadata['accuracy']*100:.2f}%",
            'precision': f"{model_metadata['precision']*100:.2f}%",
            'recall': f"{model_metadata['recall']*100:.2f}%",
            'f1_score': f"{model_metadata['f1_score']*100:.2f}%",
            'roc_auc': f"{model_metadata['roc_auc']*100:.2f}%",
            'n_features': model_metadata['n_features'],
            'training_samples': model_metadata['training_samples'],
            'test_samples': model_metadata['test_samples']
        }
        
        return jsonify(info)
    except Exception as e:
        print(f"Model info error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with current dataset"""
    try:
        print("\n" + "="*80)
        print("RETRAINING MODEL (User Requested)")
        print("="*80)
        
        if train_and_save_model():
            # Reload the new model
            if load_trained_model():
                return jsonify({
                    'success': True,
                    'message': 'Model retrained successfully',
                    'model_name': model_metadata['model_name'],
                    'accuracy': f"{model_metadata['accuracy']*100:.2f}%"
                })
        
        return jsonify({'error': 'Training failed'}), 500
        
    except Exception as e:
        print(f"Retrain error: {str(e)}")
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
    
    if load_trained_model():
        print("\n" + "="*80)
        print("✓ SERVER READY")
        print("="*80)
        print(f"\n🤖 Active Model: {model_metadata['model_name']}")
        print(f"📊 Performance:")
        print(f"   • Accuracy:  {model_metadata['accuracy']*100:.2f}%")
        print(f"   • Precision: {model_metadata['precision']*100:.2f}%")
        print(f"   • Recall:    {model_metadata['recall']*100:.2f}%")
        print(f"   • F1-Score:  {model_metadata['f1_score']*100:.2f}%")
        print(f"   • ROC-AUC:   {model_metadata['roc_auc']*100:.2f}%")
        
        print("\n🌐 Access the application at: http://127.0.0.1:5000")
        print("\n📡 Available endpoints:")
        print("   • GET  /                  (Home page)")
        print("   • POST /predict           (Single prediction)")
        print("   • POST /batch_predict     (Batch prediction)")
        print("   • GET  /analytics         (Analytics dashboard)")
        print("   • GET  /api/model_info    (Model information)")
        print("   • POST /api/retrain       (Retrain model)")
        
        print("\n💡 Tips:")
        print("   • Model automatically trains if not found")
        print("   • Use POST /api/retrain to retrain with new data")
        print("   • All predictions show which model is being used")
        
        print("\n⌨️  Press CTRL+C to stop the server")
        print("="*80 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "="*80)
        print("❌ FAILED TO START")
        print("="*80)
        print("\n⚠️  Please ensure:")
        print("   1. Dataset file exists: WA_Fn-UseC_-HR-Employee-Attrition.csv")
        print("   2. Required packages installed: pip install -r requirements.txt")
        print("\n📥 Download dataset from:")
        print("   https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print("="*80)
