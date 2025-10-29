# Career Path Analysis & Employee Attrition Prediction
# Data Mining Project using IBM HR Analytics Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class CareerPathAnalyzer:
    """
    Comprehensive Career Path Analysis System
    Uses multiple data mining techniques for employee attrition prediction
    """
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("="*80)
        print("LOADING IBM HR ANALYTICS DATASET")
        print("="*80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Employees: {len(self.df)}")
        print(f"\nColumn Names:\n{self.df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nMissing Values:")
        print(self.df.isnull().sum().sum())
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Basic statistics
        print("\n1. STATISTICAL SUMMARY")
        print(self.df.describe())
        
        # Attrition analysis
        print("\n2. ATTRITION ANALYSIS")
        attrition_counts = self.df['Attrition'].value_counts()
        attrition_pct = self.df['Attrition'].value_counts(normalize=True) * 100
        
        print(f"\nAttrition Distribution:")
        print(f"No:  {attrition_counts.get('No', 0)} ({attrition_pct.get('No', 0):.2f}%)")
        print(f"Yes: {attrition_counts.get('Yes', 0)} ({attrition_pct.get('Yes', 0):.2f}%)")
        
        # Key metrics
        print("\n3. KEY METRICS")
        print(f"Average Age: {self.df['Age'].mean():.1f} years")
        print(f"Average Monthly Income: ${self.df['MonthlyIncome'].mean():.2f}")
        print(f"Average Years at Company: {self.df['YearsAtCompany'].mean():.1f} years")
        print(f"Average Job Satisfaction: {self.df['JobSatisfaction'].mean():.2f}/4")
        
        # Department analysis
        print("\n4. DEPARTMENT-WISE ATTRITION")
        dept_attrition = pd.crosstab(self.df['Department'], 
                                      self.df['Attrition'], 
                                      normalize='index') * 100
        print(dept_attrition)
        
        # Job role analysis
        print("\n5. TOP 5 JOB ROLES WITH HIGHEST ATTRITION")
        role_attrition = self.df[self.df['Attrition']=='Yes']['JobRole'].value_counts().head()
        print(role_attrition)
        
        return self.generate_visualizations()
    
    def generate_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n6. GENERATING VISUALIZATIONS...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Attrition Distribution
        plt.subplot(3, 3, 1)
        attrition_counts = self.df['Attrition'].value_counts()
        plt.pie(attrition_counts.values, labels=attrition_counts.index, 
                autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
        plt.title('Attrition Distribution', fontsize=14, fontweight='bold')
        
        # 2. Age Distribution by Attrition
        plt.subplot(3, 3, 2)
        for attrition in self.df['Attrition'].unique():
            subset = self.df[self.df['Attrition'] == attrition]
            plt.hist(subset['Age'], alpha=0.6, label=attrition, bins=20)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Attrition', fontsize=14, fontweight='bold')
        plt.legend()
        
        # 3. Monthly Income by Attrition
        plt.subplot(3, 3, 3)
        sns.boxplot(data=self.df, x='Attrition', y='MonthlyIncome', palette='Set2')
        plt.title('Monthly Income by Attrition', fontsize=14, fontweight='bold')
        
        # 4. Job Satisfaction by Attrition
        plt.subplot(3, 3, 4)
        satisfaction_data = pd.crosstab(self.df['JobSatisfaction'], 
                                        self.df['Attrition'], 
                                        normalize='index') * 100
        satisfaction_data.plot(kind='bar', stacked=False, ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
        plt.xlabel('Job Satisfaction Level')
        plt.ylabel('Percentage')
        plt.title('Job Satisfaction vs Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        
        # 5. Years at Company Distribution
        plt.subplot(3, 3, 5)
        for attrition in self.df['Attrition'].unique():
            subset = self.df[self.df['Attrition'] == attrition]
            plt.hist(subset['YearsAtCompany'], alpha=0.6, label=attrition, bins=20)
        plt.xlabel('Years at Company')
        plt.ylabel('Frequency')
        plt.title('Tenure Distribution by Attrition', fontsize=14, fontweight='bold')
        plt.legend()
        
        # 6. Department-wise Attrition
        plt.subplot(3, 3, 6)
        dept_data = pd.crosstab(self.df['Department'], self.df['Attrition'])
        dept_data.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
        plt.xlabel('Department')
        plt.ylabel('Count')
        plt.title('Department-wise Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=45)
        
        # 7. Overtime Impact
        plt.subplot(3, 3, 7)
        overtime_data = pd.crosstab(self.df['OverTime'], 
                                     self.df['Attrition'], 
                                     normalize='index') * 100
        overtime_data.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
        plt.xlabel('Overtime')
        plt.ylabel('Percentage')
        plt.title('Overtime Impact on Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        
        # 8. Work-Life Balance
        plt.subplot(3, 3, 8)
        wlb_data = pd.crosstab(self.df['WorkLifeBalance'], 
                               self.df['Attrition'], 
                               normalize='index') * 100
        wlb_data.plot(kind='bar', stacked=False, ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
        plt.xlabel('Work-Life Balance Rating')
        plt.ylabel('Percentage')
        plt.title('Work-Life Balance vs Attrition', fontsize=14, fontweight='bold')
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        
        # 9. Correlation Heatmap (numeric features)
        plt.subplot(3, 3, 9)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        top_features = corr_matrix.nlargest(10, 'Age')['Age'].index
        sns.heatmap(self.df[top_features].corr(), annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('career_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("✓ Visualizations saved as 'career_analysis_visualizations.png'")
        plt.show()
        
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        print("\n" + "="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        # Create a copy
        df_processed = self.df.copy()
        
        # Remove unnecessary columns
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)
        print(f"\n✓ Removed unnecessary columns: {cols_to_drop}")
        
        # Encode target variable
        df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        
        print(f"\n✓ Encoding categorical variables: {list(categorical_cols)}")
        for col in categorical_cols:
            df_processed[col] = le.fit_transform(df_processed[col])
        
        # Separate features and target
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        print(f"\n✓ Feature shape: {X.shape}")
        print(f"✓ Target distribution:")
        print(f"  - No Attrition (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")
        print(f"  - Attrition (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.2f}%)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        
        # Handle class imbalance with SMOTE
        print(f"\n✓ Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"✓ After SMOTE - Train set: {self.X_train.shape[0]} samples")
        print(f"  - No Attrition (0): {(self.y_train==0).sum()}")
        print(f"  - Attrition (1): {(self.y_train==1).sum()}")
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        print(f"\n✓ Features standardized")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"\n→ Training {name}...", end=' ')
            model.fit(self.X_train, self.y_train)
            print("✓ Done")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='accuracy')
            print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"{name.upper()}")
            print('='*60)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            print(f"\nAccuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            if y_pred_proba is not None:
                auc = roc_auc_score(self.y_test, y_pred_proba)
                print(f"ROC-AUC:   {auc:.4f}")
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc if y_pred_proba is not None else None,
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
    
    def visualize_model_comparison(self):
        """Visualize model performance comparison"""
        print("\n" + "="*80)
        print("MODEL COMPARISON VISUALIZATIONS")
        print("="*80)
        
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Model Comparison - Bar Chart
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
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 3. Confusion Matrices
        for idx, (name, results) in enumerate(self.results.items()):
            plt.subplot(2, 3, idx + 3)
            cm = confusion_matrix(self.y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No', 'Yes'], 
                       yticklabels=['No', 'Yes'])
            plt.title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Model comparison visualizations saved as 'model_comparison.png'")
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance from Random Forest"""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get Random Forest model
        rf_model = self.models['Random Forest']
        
        # Get feature names (before scaling)
        df_processed = self.df.copy()
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        feature_names = [col for col in df_processed.columns if col not in cols_to_drop]
        
        # Get feature importances
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 15 Most Important Features:")
        print("-" * 60)
        for i, idx in enumerate(indices[:15], 1):
            print(f"{i:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")
        
        # Visualization
        plt.figure(figsize=(12, 8))
        top_n = 15
        top_indices = indices[:top_n]
        plt.barh(range(top_n), importances[top_indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance Score')
        plt.title('Top 15 Most Important Features for Attrition Prediction', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved as 'feature_importance.png'")
        plt.show()
    
    def employee_clustering(self):
        """Perform K-Means clustering to identify employee segments"""
        print("\n" + "="*80)
        print("EMPLOYEE CLUSTERING ANALYSIS")
        print("="*80)
        
        # Select key features for clustering
        cluster_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                          'JobSatisfaction', 'WorkLifeBalance']
        
        df_cluster = self.df[cluster_features].copy()
        
        # Standardize features
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_cluster)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, 11)
        
        print("\nDetermining optimal number of clusters...")
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform clustering with optimal k (let's use 4)
        optimal_k = 4
        print(f"✓ Using {optimal_k} clusters")
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_scaled)
        
        # Add cluster labels to dataframe
        self.df['Cluster'] = clusters
        
        # Analyze clusters
        print("\nCluster Analysis:")
        print("="*60)
        
        for i in range(optimal_k):
            cluster_data = self.df[self.df['Cluster'] == i]
            attrition_rate = (cluster_data['Attrition'] == 'Yes').sum() / len(cluster_data) * 100
            
            print(f"\nCluster {i+1} ({len(cluster_data)} employees, {attrition_rate:.1f}% attrition):")
            print(f"  Avg Age: {cluster_data['Age'].mean():.1f} years")
            print(f"  Avg Income: ${cluster_data['MonthlyIncome'].mean():.2f}")
            print(f"  Avg Tenure: {cluster_data['YearsAtCompany'].mean():.1f} years")
            print(f"  Avg Job Satisfaction: {cluster_data['JobSatisfaction'].mean():.2f}/4")
            print(f"  Avg Work-Life Balance: {cluster_data['WorkLifeBalance'].mean():.2f}/4")
        
        # Visualization
        fig = plt.figure(figsize=(18, 6))
        
        # Elbow plot
        plt.subplot(1, 3, 1)
        plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # Cluster scatter plot
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(self.df['Age'], self.df['MonthlyIncome'], 
                            c=clusters, cmap='viridis', alpha=0.6, s=50)
        plt.xlabel('Age')
        plt.ylabel('Monthly Income')
        plt.title('Employee Clusters (Age vs Income)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        
        # Cluster distribution
        plt.subplot(1, 3, 3)
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        plt.bar(range(optimal_k), cluster_counts.values, color='steelblue')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Employees')
        plt.title('Employee Distribution Across Clusters', fontsize=14, fontweight='bold')
        plt.xticks(range(optimal_k), [f'C{i+1}' for i in range(optimal_k)])
        
        plt.tight_layout()
        plt.savefig('employee_clustering.png', dpi=300, bbox_inches='tight')
        print("\n✓ Clustering visualizations saved as 'employee_clustering.png'")
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("FINAL REPORT SUMMARY")
        print("="*80)
        
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\n✓ BEST PERFORMING MODEL: {best_model[0]}")
        print(f"  - Accuracy:  {best_model[1]['accuracy']:.4f}")
        print(f"  - Precision: {best_model[1]['precision']:.4f}")
        print(f"  - Recall:    {best_model[1]['recall']:.4f}")
        print(f"  - F1-Score:  {best_model[1]['f1_score']:.4f}")
        if best_model[1]['auc']:
            print(f"  - ROC-AUC:   {best_model[1]['auc']:.4f}")
        
        print("\n✓ KEY INSIGHTS:")
        print(f"  - Total employees analyzed: {len(self.df)}")
        
        attrition_rate = (self.df['Attrition'] == 'Yes').sum() / len(self.df) * 100
        print(f"  - Overall attrition rate: {attrition_rate:.2f}%")
        
        print(f"  - Average employee age: {self.df['Age'].mean():.1f} years")
        print(f"  - Average tenure: {self.df['YearsAtCompany'].mean():.1f} years")
        print(f"  - Average monthly income: ${self.df['MonthlyIncome'].mean():.2f}")
        
        print("\n✓ RECOMMENDATIONS:")
        print("  1. Focus retention efforts on high-risk employee segments")
        print("  2. Improve job satisfaction and work-life balance initiatives")
        print("  3. Address compensation concerns, especially for lower-income groups")
        print("  4. Monitor overtime patterns and their impact on attrition")
        print("  5. Implement targeted interventions based on cluster analysis")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - career_analysis_visualizations.png")
        print("  - model_comparison.png")
        print("  - feature_importance.png")
        print("  - employee_clustering.png")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CAREER PATH ANALYSIS & EMPLOYEE ATTRITION PREDICTION")
    print("Data Mining Project")
    print("="*80)
    
    # Initialize analyzer
    # NOTE: Replace with your actual dataset path
    data_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    
    analyzer = CareerPathAnalyzer(data_path)
    
    try:
        # Step 1: Load data
        analyzer.load_data()
        
        # Step 2: EDA
        analyzer.exploratory_data_analysis()
        
        # Step 3: Preprocess
        analyzer.preprocess_data()
        
        # Step 4: Train models
        analyzer.train_models()
        
        # Step 5: Evaluate models
        analyzer.evaluate_models()
        
        # Step 6: Visualize comparison
        analyzer.visualize_model_comparison()
        
        # Step 7: Feature importance
        analyzer.feature_importance_analysis()
        
        # Step 8: Clustering
        analyzer.employee_clustering()
        
        # Step 9: Generate report
        analyzer.generate_report()
        
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