"""
Recipe Site Traffic Prediction - Streamlit App
DataCamp Data Scientist Professional Certification
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import re
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Recipe Site Traffic Predictor",
    page_icon="🍳",
    layout="wide"
)

# Title
st.title("🍳 Recipe Site Traffic Prediction")
st.markdown("### Predict which recipes will lead to high website traffic")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('workspace/recipe_site_traffic_2212.csv')
    return df

@st.cache_data
def prepare_data(df):
    df_clean = df.copy()
    
    # Clean servings column
    def clean_servings(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        numbers = re.findall(r'\d+', val_str)
        if numbers:
            return int(numbers[0])
        return np.nan
    
    df_clean['servings'] = df_clean['servings'].apply(clean_servings)
    
    # Convert target to binary
    df_clean['high_traffic'] = df_clean['high_traffic'].apply(lambda x: 1 if x == 'High' else 0)
    
    # Impute missing values
    nutritional_cols = ['calories', 'carbohydrate', 'sugar', 'protein']
    for col in nutritional_cols:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
    
    return df_clean

@st.cache_resource
def train_model(df_clean):
    # Prepare features
    X = pd.get_dummies(df_clean[['calories', 'carbohydrate', 'sugar', 'protein', 'servings', 'category']], 
                       columns=['category'], drop_first=True)
    y = df_clean['high_traffic']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return model, scaler, X.columns.tolist(), metrics, X_test_scaled, y_test, y_pred

# Load data
try:
    df = load_data()
    df_clean = prepare_data(df)
    model, scaler, feature_cols, metrics, X_test_scaled, y_test, y_pred = train_model(df_clean)
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if data_loaded:
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Model Performance", "Make Predictions"])
    
    if page == "Overview":
        st.header("📋 Project Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Recipes", len(df_clean))
        with col2:
            st.metric("High Traffic %", f"{df_clean['high_traffic'].mean()*100:.1f}%")
        with col3:
            st.metric("Categories", df_clean['category'].nunique())
        with col4:
            st.metric("Features", 6)
        
        st.markdown("""
        ### 🎯 Business Goal
        Build a model to predict which recipes will lead to high website traffic with **80%+ accuracy** 
        to help Product Managers decide which recipes to feature on the homepage.
        
        ### 📊 Dataset Features
        - **Nutritional**: Calories, Carbohydrate, Sugar, Protein
        - **Recipe Info**: Category, Servings
        - **Target**: High Traffic (Yes/No)
        """)
        
    elif page == "Data Exploration":
        st.header("🔍 Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Distribution", "Categories", "Correlations"])
        
        with tab1:
            st.subheader("Target Variable Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            traffic_counts = df_clean['high_traffic'].value_counts()
            colors = ['#ff6b6b', '#4ecdc4']
            ax.bar(['Low Traffic', 'High Traffic'], traffic_counts.values, color=colors)
            ax.set_ylabel('Number of Recipes')
            ax.set_title('Distribution of Traffic Types')
            for i, v in enumerate(traffic_counts.values):
                ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Traffic by Category")
            fig, ax = plt.subplots(figsize=(12, 6))
            category_traffic = df_clean.groupby('category')['high_traffic'].mean().sort_values(ascending=False)
            bars = ax.bar(category_traffic.index, category_traffic.values * 100, color='steelblue')
            ax.axhline(y=80, color='red', linestyle='--', label='80% Target')
            ax.set_ylabel('High Traffic %')
            ax.set_xlabel('Category')
            ax.set_title('High Traffic Rate by Category')
            plt.xticks(rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("📌 Key Insight: Vegetable, Potato, and Pork categories have the highest traffic rates!")
            
        with tab3:
            st.subheader("Feature Correlations")
            fig, ax = plt.subplots(figsize=(10, 8))
            numeric_cols = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings', 'high_traffic']
            corr_matrix = df_clean[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            
    elif page == "Model Performance":
        st.header("📈 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision']*100:.1f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall']*100:.1f}%")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']*100:.1f}%")
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted Low', 'Predicted High'],
                    yticklabels=['Actual Low', 'Actual High'])
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Low Traffic', 'High Traffic'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
    elif page == "Make Predictions":
        st.header("🔮 Predict Recipe Traffic")
        st.markdown("Enter recipe details to predict if it will generate high traffic:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            calories = st.number_input("Calories", min_value=0.0, max_value=5000.0, value=300.0, step=10.0)
            carbohydrate = st.number_input("Carbohydrate (g)", min_value=0.0, max_value=600.0, value=20.0, step=1.0)
            sugar = st.number_input("Sugar (g)", min_value=0.0, max_value=150.0, value=5.0, step=0.1)
            
        with col2:
            protein = st.number_input("Protein (g)", min_value=0.0, max_value=400.0, value=10.0, step=1.0)
            servings = st.selectbox("Servings", [1, 2, 4, 6])
            category = st.selectbox("Category", sorted(df_clean['category'].unique()))
        
        if st.button("🎯 Predict Traffic", type="primary"):
            # Prepare input
            input_data = pd.DataFrame({
                'calories': [calories],
                'carbohydrate': [carbohydrate],
                'sugar': [sugar],
                'protein': [protein],
                'servings': [servings],
                'category': [category]
            })
            
            # One-hot encode
            input_encoded = pd.get_dummies(input_data, columns=['category'], drop_first=True)
            
            # Add missing columns
            for col in feature_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns
            input_encoded = input_encoded[feature_cols]
            
            # Scale and predict
            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success(f"### ✅ HIGH TRAFFIC Predicted!")
                st.balloons()
            else:
                st.warning(f"### ⚠️ LOW TRAFFIC Predicted")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Low Traffic Probability", f"{probability[0]*100:.1f}%")
            with col2:
                st.metric("High Traffic Probability", f"{probability[1]*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Nagendra Singh Rawat | [GitHub](https://github.com/iNSRawat)")
