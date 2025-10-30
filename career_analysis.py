"""
Train and Save Best Model for Career Path Analysis
This script trains all models, selects the best one, saves it, and generates visualizations
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ModelTrainer:
    """Train, evaluate, and save the best attrition prediction model"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_original = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_preprocess(self):
        """Load data and preprocess"""
        print("="*80)
        print("LOADING AND PREPROCESSING DATA")
        print("="*80)
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        self.df_original = self.df.copy()  # Keep original for visualizations
        print(f"\n✓ Loaded dataset: {self.df.shape}")
        print(f"  Total employees: {len(self.df)}")
        
        # Store original data for web app
        df_processed = self.df.copy()
        
        # Drop unnecessary columns
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)
        print(f"✓ Dropped columns: {cols_to_drop}")
        
        # Encode target variable
        df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables and store encoders
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        print(f"✓ Encoding {len(categorical_cols)} categorical columns")
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        self.feature_names = X.columns.tolist()
        print(f"✓ Total features: {len(self.feature_names)}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        
        # Class distribution
        print(f"\nClass distribution in training:")
        print(f"  No Attrition (0): {(self.y_train==0).sum()} ({(self.y_train==0).sum()/len(self.y_train)*100:.1f}%)")
        print(f"  Attrition (1): {(self.y_train==1).sum()} ({(self.y_train==1).sum()/len(self.y_train)*100:.1f}%)")
        
        # Apply SMOTE
        print(f"\n✓ Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"  After SMOTE:")
        print(f"  No Attrition (0): {(self.y_train==0).sum()}")
        print(f"  Attrition (1): {(self.y_train==1).sum()}")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(f"✓ Features standardized\n")
        
    def train_all_models(self):
        """Train all candidate models"""
        print("="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in self.models.items():
            print(f"\n→ Training {name}...")
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='accuracy')
            print(f"  Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
    def evaluate_all_models(self):
        """Evaluate all models and select the best one"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        best_score = 0
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"{name.upper()}")
            print('='*60)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            print(f"\nAccuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            if y_pred_proba is not None:
                print(f"ROC-AUC:   {auc:.4f}")
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Classification report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['No Attrition', 'Attrition']))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"                 No    Yes")
            print(f"Actual No    {cm[0][0]:5d} {cm[0][1]:5d}")
            print(f"       Yes   {cm[1][0]:5d} {cm[1][1]:5d}")
            
            # Track best model based on composite score
            composite_score = f1 * 0.4 + auc * 0.4 + accuracy * 0.2
            
            if composite_score > best_score:
                best_score = composite_score
                self.best_model = model
                self.best_model_name = name
        
        # Print best model
        print("\n" + "="*80)
        print("BEST MODEL SELECTION")
        print("="*80)
        print(f"\n✓ BEST MODEL: {self.best_model_name}")
        print(f"\n  Performance Metrics:")
        print(f"  - Accuracy:  {self.results[self.best_model_name]['accuracy']:.4f}")
        print(f"  - Precision: {self.results[self.best_model_name]['precision']:.4f}")
        print(f"  - Recall:    {self.results[self.best_model_name]['recall']:.4f}")
        print(f"  - F1-Score:  {self.results[self.best_model_name]['f1_score']:.4f}")
        print(f"  - ROC-AUC:   {self.results[self.best_model_name]['auc']:.4f}")
        
        print(f"\n  Selection Criteria:")
        print(f"  - Composite score: F1(40%) + AUC(40%) + Accuracy(20%)")
        print(f"  - Best balance between precision and recall")
        
    def generate_eda_visualizations(self):
        """Generate exploratory data analysis visualizations"""
        print("\n" + "="*80)
        print("GENERATING EDA VISUALIZATIONS")
        print("="*80)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Attrition Distribution
        plt.subplot(3, 3, 1)
        attrition_counts = self.df_original['Attrition'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        plt.pie(attrition_counts.values, labels=attrition_counts.index, 
                autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Attrition Distribution', fontsize=14, fontweight='bold')
        
        # 2. Age Distribution by Attrition
        plt.subplot(3, 3, 2)
        for attrition in self.df_original['Attrition'].unique():
            subset = self.df_original[self.df_original['Attrition'] == attrition]
            plt.hist(subset['Age'], alpha=0.6, label=attrition, bins=20)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Attrition', fontsize=14, fontweight='bold')
        plt.legend()
        
        # 3. Monthly Income by Attrition
        plt.subplot(3, 3, 3)
        sns.boxplot(data=self.df_original, x='Attrition', y='MonthlyIncome', palette='Set2')
        plt.title('Monthly Income by Attrition', fontsize=14, fontweight='bold')
        
        # 4. Job Satisfaction vs Attrition
        plt.subplot(3, 3, 4)
        satisfaction_data = pd.crosstab(self.df_original['JobSatisfaction'], 
                                        self.df_original['Attrition'], 
                                        normalize='index') * 100
        satisfaction_data.plot(kind='bar', stacked=False, ax=plt.gca(), color=colors)
        plt.xlabel('Job Satisfaction Level')
        plt.ylabel('Percentage')
        plt.title('Job Satisfaction vs Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        
        # 5. Years at Company Distribution
        plt.subplot(3, 3, 5)
        for attrition in self.df_original['Attrition'].unique():
            subset = self.df_original[self.df_original['Attrition'] == attrition]
            plt.hist(subset['YearsAtCompany'], alpha=0.6, label=attrition, bins=20)
        plt.xlabel('Years at Company')
        plt.ylabel('Frequency')
        plt.title('Tenure Distribution by Attrition', fontsize=14, fontweight='bold')
        plt.legend()
        
        # 6. Department-wise Attrition
        plt.subplot(3, 3, 6)
        dept_data = pd.crosstab(self.df_original['Department'], self.df_original['Attrition'])
        dept_data.plot(kind='bar', stacked=True, ax=plt.gca(), color=colors)
        plt.xlabel('Department')
        plt.ylabel('Count')
        plt.title('Department-wise Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=45)
        
        # 7. Overtime Impact
        plt.subplot(3, 3, 7)
        overtime_data = pd.crosstab(self.df_original['OverTime'], 
                                     self.df_original['Attrition'], 
                                     normalize='index') * 100
        overtime_data.plot(kind='bar', ax=plt.gca(), color=colors)
        plt.xlabel('Overtime')
        plt.ylabel('Percentage')
        plt.title('Overtime Impact on Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        
        # 8. Work-Life Balance
        plt.subplot(3, 3, 8)
        wlb_data = pd.crosstab(self.df_original['WorkLifeBalance'], 
                               self.df_original['Attrition'], 
                               normalize='index') * 100
        wlb_data.plot(kind='bar', stacked=False, ax=plt.gca(), color=colors)
        plt.xlabel('Work-Life Balance Rating')
        plt.ylabel('Percentage')
        plt.title('Work-Life Balance vs Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        
        # 9. Correlation Heatmap
        plt.subplot(3, 3, 9)
        numeric_cols = self.df_original.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df_original[numeric_cols].corr()
        top_features = corr_matrix.nlargest(10, 'Age')['Age'].index
        sns.heatmap(self.df_original[top_features].corr(), annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Create outputs directory if it doesn't exist
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
            
        plt.savefig('outputs/career_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/career_analysis_visualizations.png")
        plt.close()
        
    def generate_model_comparison_visualizations(self):
        """Generate model comparison visualizations"""
        print("\n" + "="*80)
        print("GENERATING MODEL COMPARISON VISUALIZATIONS")
        print("="*80)
        
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Model Comparison Bar Chart
        plt.subplot(2, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(self.results))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in self.results]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width*1.5, list(self.results.keys()), rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # 2. ROC Curves
        plt.subplot(2, 3, 2)
        for name, results in self.results.items():
            if results['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
                auc = results['auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        
        # 3-6. Confusion Matrices
        for idx, (name, results) in enumerate(self.results.items()):
            plt.subplot(2, 3, idx + 3)
            cm = confusion_matrix(self.y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No', 'Yes'], 
                       yticklabels=['No', 'Yes'],
                       cbar=False)
            plt.title(f'{name}\nConfusion Matrix', fontsize=11, fontweight='bold')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/model_comparison.png")
        plt.close()
        
    def generate_feature_importance_visualization(self):
        """Generate feature importance visualization"""
        print("\n" + "="*80)
        print("GENERATING FEATURE IMPORTANCE VISUALIZATION")
        print("="*80)
        
        # Check if best model has feature importances
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            plt.figure(figsize=(12, 8))
            top_n = 15
            plt.barh(range(top_n), importances[indices], color='steelblue')
            plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top 15 Most Important Features - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: outputs/feature_importance.png")
            plt.close()
            
            # Print top features
            print("\nTop 15 Features:")
            for i, idx in enumerate(indices, 1):
                print(f"  {i:2d}. {self.feature_names[idx]:30s} : {importances[idx]:.4f}")
                
        elif hasattr(self.best_model, 'coef_'):
            # For Logistic Regression
            coef = np.abs(self.best_model.coef_[0])
            indices = np.argsort(coef)[::-1][:15]
            
            plt.figure(figsize=(12, 8))
            top_n = 15
            plt.barh(range(top_n), coef[indices], color='steelblue')
            plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
            plt.xlabel('Coefficient Magnitude', fontsize=12)
            plt.title(f'Top 15 Most Important Features - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: outputs/feature_importance.png")
            plt.close()
            
            # Print top features
            print("\nTop 15 Features:")
            for i, idx in enumerate(indices, 1):
                print(f"  {i:2d}. {self.feature_names[idx]:30s} : {coef[idx]:.4f}")
        else:
            print("⚠️  Model doesn't support feature importance")
    
    def generate_clustering_visualization(self):
        """Generate employee clustering visualization"""
        print("\n" + "="*80)
        print("GENERATING CLUSTERING VISUALIZATION")
        print("="*80)
        
        # Select features for clustering
        cluster_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                          'JobSatisfaction', 'WorkLifeBalance']
        
        df_cluster = self.df_original[cluster_features].copy()
        
        # Standardize
        scaler_cluster = StandardScaler()
        df_scaled = scaler_cluster.fit_transform(df_cluster)
        
        # Elbow method
        inertias = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform clustering with k=4
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_scaled)
        
        # Add cluster labels
        self.df_original['Cluster'] = clusters
        
        # Visualization
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Elbow plot
        plt.subplot(1, 3, 1)
        plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=11)
        plt.ylabel('Inertia (WCSS)', fontsize=11)
        plt.title('Elbow Method for Optimal k', fontsize=13, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # 2. Cluster scatter plot
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(self.df_original['Age'], 
                             self.df_original['MonthlyIncome'], 
                             c=clusters, cmap='viridis', alpha=0.6, s=50)
        plt.xlabel('Age', fontsize=11)
        plt.ylabel('Monthly Income', fontsize=11)
        plt.title('Employee Clusters (Age vs Income)', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        
        # 3. Cluster distribution
        plt.subplot(1, 3, 3)
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        plt.bar(range(optimal_k), cluster_counts.values, color='steelblue')
        plt.xlabel('Cluster', fontsize=11)
        plt.ylabel('Number of Employees', fontsize=11)
        plt.title('Employee Distribution Across Clusters', fontsize=13, fontweight='bold')
        plt.xticks(range(optimal_k), [f'C{i+1}' for i in range(optimal_k)])
        
        plt.tight_layout()
        plt.savefig('outputs/employee_clustering.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: outputs/employee_clustering.png")
        plt.close()
        
        # Print cluster analysis
        print("\nCluster Analysis:")
        for i in range(optimal_k):
            cluster_data = self.df_original[self.df_original['Cluster'] == i]
            attrition_rate = (cluster_data['Attrition'] == 'Yes').sum() / len(cluster_data) * 100
            
            print(f"\n  Cluster {i+1} ({len(cluster_data)} employees, {attrition_rate:.1f}% attrition):")
            print(f"    Avg Age: {cluster_data['Age'].mean():.1f} years")
            print(f"    Avg Income: ${cluster_data['MonthlyIncome'].mean():.2f}")
            print(f"    Avg Tenure: {cluster_data['YearsAtCompany'].mean():.1f} years")
    
    def save_model(self):
        """Save the best model and preprocessing objects"""
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save best model
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Saved best model: models/best_model.pkl")
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler: models/scaler.pkl")
        
        # Save feature names
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✓ Saved feature names: models/feature_names.pkl")
        
        # Save label encoders
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"✓ Saved label encoders: models/label_encoders.pkl")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'accuracy': self.results[self.best_model_name]['accuracy'],
            'precision': self.results[self.best_model_name]['precision'],
            'recall': self.results[self.best_model_name]['recall'],
            'f1_score': self.results[self.best_model_name]['f1_score'],
            'roc_auc': self.results[self.best_model_name]['auc'],
            'n_features': len(self.feature_names),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Saved metadata: models/model_metadata.pkl")
        
        print(f"\n✓ All model files saved successfully!")
        
    def generate_summary(self):
        """Generate final summary"""
        print("\n" + "="*80)
        print("FINAL REPORT SUMMARY")
        print("="*80)
        
        best = self.results[self.best_model_name]
        
        print(f"\n✓ BEST PERFORMING MODEL: {self.best_model_name}")
        print(f"  - Accuracy:  {best['accuracy']:.4f}")
        print(f"  - Precision: {best['precision']:.4f}")
        print(f"  - Recall:    {best['recall']:.4f}")
        print(f"  - F1-Score:  {best['f1_score']:.4f}")
        print(f"  - ROC-AUC:   {best['auc']:.4f}")
        
        print(f"\n✓ KEY INSIGHTS:")
        print(f"  - Total employees analyzed: {len(self.df_original)}")
        
        attrition_rate = (self.df_original['Attrition'] == 'Yes').sum() / len(self.df_original) * 100
        print(f"  - Overall attrition rate: {attrition_rate:.2f}%")
        print(f"  - Average employee age: {self.df_original['Age'].mean():.1f} years")
        print(f"  - Average tenure: {self.df_original['YearsAtCompany'].mean():.1f} years")
        print(f"  - Average monthly income: ${self.df_original['MonthlyIncome'].mean():.2f}")
        
        print(f"\n✓ RECOMMENDATIONS:")
        print(f"  1. Focus retention efforts on high-risk employee segments")
        print(f"  2. Improve job satisfaction and work-life balance initiatives")
        print(f"  3. Address compensation concerns, especially for lower-income groups")
        print(f"  4. Monitor overtime patterns and their impact on attrition")
        print(f"  5. Implement targeted interventions based on model predictions")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print("\nGenerated Files:")
        print("  Models:")
        print("    - models/best_model.pkl")
        print("    - models/scaler.pkl")
        print("    - models/feature_names.pkl")
        print("    - models/label_encoders.pkl")
        print("    - models/model_metadata.pkl")
        print("  Visualizations:")
        print("    - outputs/career_analysis_visualizations.png")
        print("    - outputs/model_comparison.png")
        print("    - outputs/feature_importance.png")
        print("    - outputs/employee_clustering.png")
        print("\nYou can now run 'python app.py' to start the web application")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CAREER PATH ANALYSIS - MODEL TRAINING")
    print("="*80)
    
    # Dataset path
    data_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(data_path)
        
        # Step 1: Load and preprocess
        trainer.load_and_preprocess()
        
        # Step 2: Train all models
        trainer.train_all_models()
        
        # Step 3: Evaluate and select best
        trainer.evaluate_all_models()
        
        # Step 4: Generate EDA visualizations
        trainer.generate_eda_visualizations()
        
        # Step 5: Generate model comparison visualizations
        trainer.generate_model_comparison_visualizations()
        
        # Step 6: Generate feature importance visualization
        trainer.generate_feature_importance_visualization()
        
        # Step 7: Generate clustering visualization
        trainer.generate_clustering_visualization()
        
        # Step 8: Save best model
        trainer.save_model()
        
        # Step 9: Generate summary
        trainer.generate_summary()
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Dataset file '{data_path}' not found!")
        print("\nPlease download the IBM HR Analytics dataset from:")
        print("https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print("\nSave it as 'WA_Fn-UseC_-HR-Employee-Attrition.csv' in the same directory.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
