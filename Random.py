import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Cancer Prediction Dashboard", layout="wide")

st.markdown('<h1 style="text-align: center; color: #1f77b4;">🏥 Cancer Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose cancerdata.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Patients", len(df))
    col2.metric("Cancer Cases", df['diagnosis'].sum())
    
    # Dataset preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    
    # Prepare data - Use EXACT column names
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    with st.spinner("Training models..."):
        # Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(x_train, y_train)
        dt_pred = dt_model.predict(x_test)
        
        # Random Forest (your exact params)
        rf_model = RandomForestClassifier(
            n_estimators=25,
            max_features='log2',
            criterion='gini',
            bootstrap=False,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        rf_model.fit(x_train, y_train)
        rf_pred = rf_model.predict(x_test)
    
    # Model comparison
    st.header("🤖 Model Results")
    col1, col2 = st.columns(2)
    col1.metric("Decision Tree Accuracy", f"{dt_model.score(x_test, y_test):.3f}")
    col2.metric("Random Forest Accuracy", f"{rf_model.score(x_test, y_test):.3f}")
    
    # Confusion matrices
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Decision Tree', 'Random Forest'))
    fig.add_trace(go.Bar(x=['Correct', 'Wrong'], y=confusion_matrix(y_test, dt_pred).flatten(),
                        name='DT'), row=1, col=1)
    fig.add_trace(go.Bar(x=['Correct', 'Wrong'], y=confusion_matrix(y_test, rf_pred).flatten(),
                        name='RF'), row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig_bar = px.bar(importances.tail(8), x='importance', y='feature', orientation='h')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Prediction section - FIXED COLUMN NAMES
    st.header("🔮 Predict New Patient")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 20, 80, 50)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    
    with col2:
        bmi = st.slider("BMI", 15.0, 40.0, 25.0)
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col3:
        genetic_risk = st.selectbox("Genetic Risk", [0, 1, 2], 
                                  format_func=lambda x: ["Low", "Medium", "High"][x])
        physical_activity = st.slider("Physical Activity (hrs/wk)", 0.0, 10.0, 5.0)
    
    col1, col2 = st.columns(2)
    with col1:
        alcohol_intake = st.slider("Alcohol Intake (units/wk)", 0.0, 5.0, 2.0)
        cancer_history = st.selectbox("Cancer History", [0, 1], 
                                    format_func=lambda x: "No" if x == 0 else "Yes")
    
    if st.button("🔍 Predict", type="primary"):
        # FIXED: Use EXACT column names from CSV (snake_case)
        new_patient = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'bmi': [bmi],
            'smoking': [smoking],
            'genetic_risk': [genetic_risk],      # ✅ Fixed
            'physical_activity': [physical_activity],  # ✅ Fixed
            'alcohol_intake': [alcohol_intake],  # ✅ Fixed
            'cancer_history': [cancer_history]   # ✅ Fixed
        })
        
        dt_prob = dt_model.predict_proba(new_patient)[0][1]
        rf_prob = rf_model.predict_proba(new_patient)[0][1]
        
        st.success(f"Decision Tree: {dt_prob:.1%} cancer risk")
        st.error(f"Random Forest: {rf_prob:.1%} cancer risk")
        
        risk = "HIGH" if rf_prob > 0.7 else "MEDIUM" if rf_prob > 0.3 else "LOW"
        st.markdown(f"**Risk Level: {risk}**")

else:
    st.info("👈 Upload cancerdata.csv")
