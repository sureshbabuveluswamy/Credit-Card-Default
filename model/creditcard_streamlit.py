import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import requests
import io

# Set page configuration
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and Objective
st.markdown('<h1 class="main-header">🏦 Credit Card Default Prediction</h1>', unsafe_allow_html=True)

st.markdown("""
### 📊 Project Objective

This project focuses on predicting credit card default payments using various machine learning algorithms. 
The dataset contains information about credit card clients in Taiwan from April 2005 to September 2005, 
and we aim to predict which clients will default on their payments in the following month.

**Key Features:**
- Multiple ML models for comparison
- Real-time predictions on uploaded data
- Model performance visualization
- Download test datasets
""")

# Load models from folder
@st.cache_data
def load_models():
    # Use current file's directory for cross-platform compatibility
    models_dir = os.path.dirname(os.path.abspath(__file__))
    models = {}
    
    model_files = {
        'Logistic Regression': 'LogisticRegression.pkl',
        'Decision Tree': 'DecisionTreeClassifier.pkl',
        'K-Nearest Neighbors': 'KNearestNeighborClassifier.pkl',
        'Naive Bayes': 'NaiveBayesClassifierGaussian.pkl',
        'Random Forest': 'EnsembleModelRandomForest.pkl',
        'XGBoost': 'EnsembleModelXGBoost.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            models[model_name] = joblib.load(filepath)
    
    # Load scaler and feature names
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    
    with open(os.path.join(models_dir, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    with open(os.path.join(models_dir, 'numerical_features.json'), 'r') as f:
        numerical_features = json.load(f)
    
    return models, scaler, feature_names, numerical_features

models, scaler, feature_names, numerical_features = load_models()

# Sidebar for navigation
st.sidebar.title("🔧 Navigation")

# Initialize session state for page selection
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📋 Model Overview"

# Navigation buttons
pages = ["📋 Model Overview", "📊 Model Visualization", "📥 Download Dataset", "📤 Upload & Predict", "📈 Model Evaluation", "📄 README"]

for page in pages:
    if st.sidebar.button(page, use_container_width=True):
        st.session_state.current_page = page

page = st.session_state.current_page

# Page 1: Model Overview
if page == "📋 Model Overview":
    st.markdown('<h2 class="section-header">📋 Model Overview</h2>', unsafe_allow_html=True)
    
    st.write("### Available Models") 
    
    for model_name in models.keys():
        with st.expander(f"🤖 {model_name}"):
            if model_name == "Logistic Regression":
                st.write("**Type:** Linear classification model")
                st.write("**Best for:** Binary classification with interpretable coefficients")
                st.write("**Characteristics:** Conservative predictions, high precision")
            elif model_name == "Decision Tree":
                st.write("**Type:** Tree-based classifier")
                st.write("**Best for:** Interpretable decision rules")
                st.write("**Characteristics:** Can capture non-linear patterns")
            elif model_name == "K-Nearest Neighbors":
                st.write("**Type:** Instance-based learning")
                st.write("**Best for:** Local pattern recognition")
                st.write("**Characteristics:** Distance-based predictions")
            elif model_name == "Naive Bayes":
                st.write("**Type:** Probabilistic classifier")
                st.write("**Best for:** Fast predictions with independence assumptions")
                st.write("**Characteristics:** High recall, low precision")
            elif model_name == "Random Forest":
                st.write("**Type:** Ensemble tree-based model")
                st.write("**Best for:** Reducing overfitting")
                st.write("**Characteristics:** Bagging ensemble of decision trees")
            elif model_name == "XGBoost":
                st.write("**Type:** Gradient boosting ensemble")
                st.write("**Best for:** High performance on structured data")
                st.write("**Characteristics:** Sequential tree building")

# Page 2: Model Visualization
elif page == "📊 Model Visualization":
    st.markdown('<h2 class="section-header">📊 Model Visualization</h2>', unsafe_allow_html=True)
    
    selected_model = st.selectbox("Select Model to Visualize:", list(models.keys()))
    
    if selected_model:
        model = models[selected_model]
        
        # Load sample data for visualization
        try:
            # Try to load test data from GitHub
            url = "https://raw.githubusercontent.com/sureshbabuveluswamy/Credit-Card-Default/main/Dataset/Credit_card_testdata.csv"
            sample_data = pd.read_csv(url)
            
            st.write(f"### 📊 {selected_model} - Sample Predictions")
            st.write(f"Dataset shape: {sample_data.shape}")
            
            # Prepare data for prediction
            # Apply same feature engineering as in training
            X_sample = sample_data.copy()
            
            # Feature engineering (simplified version)
            bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            
            X_sample['TOTAL_BILL_AMT'] = X_sample[bill_cols].sum(axis=1)
            X_sample['TOTAL_PAY_AMT'] = X_sample[pay_cols].sum(axis=1)
            X_sample['AVG_BILL_AMT'] = X_sample[bill_cols].mean(axis=1)
            X_sample['AVG_PAY_AMT'] = X_sample[pay_cols].mean(axis=1)
            X_sample['PAY_TO_BILL_RATIO'] = X_sample['TOTAL_PAY_AMT'] / (X_sample['TOTAL_BILL_AMT'] + 1)
            X_sample['WORST_PAYMENT_STATUS'] = X_sample[pay_status_cols].max(axis=1)
            X_sample['CREDIT_UTILIZATION'] = X_sample['BILL_AMT1'] / (X_sample['LIMIT_BAL'] + 1)
            
            # Age groups
            X_sample['AGE_GROUP'] = pd.cut(X_sample['AGE'], 
                                       bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
            
            # Drop ID and one-hot encode
            X_sample = X_sample.drop('ID', axis=1)
            X_sample = pd.get_dummies(X_sample, columns=['AGE_GROUP'], drop_first=True)
            
            # Align columns with training data
            for col in feature_names:
                if col not in X_sample.columns:
                    X_sample[col] = 0
            X_sample = X_sample[feature_names]
            
            # Scale numerical features
            X_sample[numerical_features] = scaler.transform(X_sample[numerical_features])
            
            # Make predictions
            predictions = model.predict(X_sample[:100])  # First 100 samples
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_sample[:100])
            
            # Display predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Prediction Distribution")
                pred_counts = pd.Series(predictions).value_counts()
                fig, ax = plt.subplots()
                pred_counts.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
                ax.set_xlabel('Prediction')
                ax.set_ylabel('Count')
                ax.set_title('Default vs No Default')
                st.pyplot(fig)
            
            with col2:
                st.write("#### Sample Predictions")
                sample_df = pd.DataFrame({
                    'Sample_Index': range(min(20, len(predictions))),
                    'Prediction': predictions[:20],
                    'Probability_Default': probabilities[:20, 1] if probabilities is not None else 'N/A'
                })
                st.dataframe(sample_df)
                
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Page 3: Download Dataset
elif page == "📥 Download Dataset":
    st.markdown('<h2 class="section-header">📥 Download Dataset</h2>', unsafe_allow_html=True)
    
    st.write("### Download Test Dataset")
    st.write("Download the test dataset used for model evaluation:")
    
    try:
        # Load test data from GitHub
        url = "https://raw.githubusercontent.com/sureshbabuveluswamy/Credit-Card-Default/main/Dataset/Credit_card_testdata.csv"
        test_data = pd.read_csv(url)
        
        st.write(f"**Dataset Info:**")
        st.write(f"- Shape: {test_data.shape}")
        st.write(f"- Columns: {list(test_data.columns)}")
        
        # Download button
        csv = test_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Credit_card_testdata.csv",
            data=csv,
            file_name="Credit_card_testdata.csv",
            mime="text/csv"
        )
        
        # Show preview
        st.write("#### Dataset Preview")
        st.dataframe(test_data.head())
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# Page 4: Upload and Predict
elif page == "📤 Upload & Predict":
    st.markdown('<h2 class="section-header">📤 Upload & Predict</h2>', unsafe_allow_html=True)
    
    st.write("### Upload your data for prediction")
    st.write("Upload a CSV file with the same structure as the training data:")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            data = pd.read_csv(uploaded_file)
            st.write(f"**Uploaded data shape:** {data.shape}")
            
            # Apply feature engineering
            X_upload = data.copy()
            
            # Check required columns
            required_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                          'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                          'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                          'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            missing_cols = [col for col in required_cols if col not in X_upload.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Feature engineering
                bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
                pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
                
                X_upload['TOTAL_BILL_AMT'] = X_upload[bill_cols].sum(axis=1)
                X_upload['TOTAL_PAY_AMT'] = X_upload[pay_cols].sum(axis=1)
                X_upload['AVG_BILL_AMT'] = X_upload[bill_cols].mean(axis=1)
                X_upload['AVG_PAY_AMT'] = X_upload[pay_cols].mean(axis=1)
                X_upload['PAY_TO_BILL_RATIO'] = X_upload['TOTAL_PAY_AMT'] / (X_upload['TOTAL_BILL_AMT'] + 1)
                X_upload['WORST_PAYMENT_STATUS'] = X_upload[pay_status_cols].max(axis=1)
                X_upload['CREDIT_UTILIZATION'] = X_upload['BILL_AMT1'] / (X_upload['LIMIT_BAL'] + 1)
                
                # Age groups
                X_upload['AGE_GROUP'] = pd.cut(X_upload['AGE'], 
                                           bins=[0, 25, 35, 45, 55, 100], 
                                           labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
                
                # Drop ID if exists and one-hot encode
                if 'ID' in X_upload.columns:
                    X_upload = X_upload.drop('ID', axis=1)
                X_upload = pd.get_dummies(X_upload, columns=['AGE_GROUP'], drop_first=True)
                
                # Align columns
                for col in feature_names:
                    if col not in X_upload.columns:
                        X_upload[col] = 0
                X_upload = X_upload[feature_names]
                
                # Scale numerical features
                X_upload[numerical_features] = scaler.transform(X_upload[numerical_features])
                
                st.success("Data processed successfully!")
                
                # Model selection for prediction
                selected_model = st.selectbox("Select Model for Prediction:", list(models.keys()))
                
                if selected_model and st.button("🔮 Make Predictions"):
                    model = models[selected_model]
                    predictions = model.predict(X_upload)
                    probabilities = None
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_upload)
                    
                    # Display results
                    st.write(f"### 📊 Predictions using {selected_model}")
                    
                    results_df = data.copy() if 'ID' in data.columns else pd.DataFrame()
                    results_df['Prediction'] = predictions
                    if probabilities is not None:
                        results_df['Probability_Default'] = probabilities[:, 1]
                        results_df['Probability_No_Default'] = probabilities[:, 0]
                    
                    st.dataframe(results_df)
                    
                    # Download predictions
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv_results,
                        file_name=f"predictions_{selected_model.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Predicted Defaults", sum(predictions))
                    with col3:
                        st.metric("Default Rate", f"{sum(predictions)/len(predictions)*100:.1f}%")
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Page 5: Model Evaluation
elif page == "📈 Model Evaluation":
    st.markdown('<h2 class="section-header">📈 Model Evaluation & Comparison</h2>', unsafe_allow_html=True)
    
    st.write("### Model Performance Comparison")
    
    # Performance metrics (based on training set evaluation from notebook)
    performance_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Naive Bayes', 'Random Forest', 'XGBoost'],
        'Accuracy': [80.72, 22.47, 84.40, 42.27, 67.08, 67.48],
        'AUC Score': [0.7446, 0.3819, 0.8819, 0.7385, 0.5697, 0.6117],
        'Precision': [68.99, 17.32, 72.41, 26.10, 24.64, 32.32],
        'Recall': [23.34, 66.38, 47.65, 87.93, 23.71, 42.96],
        'F1 Score': [34.88, 27.47, 57.48, 40.26, 24.17, 36.89],
        'MCC Score': [32.11, -27.29, 50.01, 16.41, 3.15, 15.86]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Display comparison table
    st.dataframe(perf_df)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(perf_df['Model'], perf_df['Accuracy'], color='skyblue')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.write("#### AUC Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(perf_df['Model'], perf_df['AUC Score'], color='lightcoral')
        ax.set_xlabel('Model')
        ax.set_ylabel('AUC Score')
        ax.set_title('Model AUC Score Comparison')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    # Detailed metrics comparison
    st.write("#### Detailed Metrics Comparison")
    
    metric = st.selectbox("Select Metric to Compare:", 
                       ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score'])
    
    if metric:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(perf_df)))
        bars = ax.bar(perf_df['Model'], perf_df[metric], color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison Across Models')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(perf_df[metric])*0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    # Model recommendations
    st.write("#### 🎯 Model Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🏆 Best Overall: K-Nearest Neighbors**
        - Highest accuracy (84.4%)
        - Best AUC score (0.882)
        - Good balance of precision and recall
        """)
    
    with col2:
        st.warning("""
        **⚠️ Conservative Choice: Logistic Regression**
        - High precision (69%)
        - Low false positive rate
        - Good for minimizing false alarms
        """)
    
    with col3:
        st.error("""
        **🔍 High Recall: Naive Bayes**
        - Highest recall (87.9%)
        - Catches most defaults
        - High false positive rate
        """)

# Page 6: README
elif page == "📄 README":
    st.markdown('<h2 class="section-header">📄 README</h2>', unsafe_allow_html=True)
    
    # Get parent directory for README
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'README.md')
    
    try:
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        st.markdown(readme_content)
    except Exception as e:
        st.error(f"Could not load README.md: {e}")
        st.info("README.md should be in the repository root directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏦 Credit Card Default Prediction System | Built with Streamlit</p>
    <p>Models trained on Taiwan Credit Card Dataset (2005)</p>
</div>
""", unsafe_allow_html=True)
