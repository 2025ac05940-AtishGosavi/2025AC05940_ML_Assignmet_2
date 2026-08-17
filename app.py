import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report

# title
st.title("Machine Learning Assignment 2")
st.write("Student ML Model Evaluator for Diabetes Dataset")

# User inputs
st.sidebar.header("User Inputs")

# a. Dataset upload option
uploaded_file = st.sidebar.file_uploader("Upload your test data (CSV)", type="csv")

# b. Model selection dropdown
model_name = st.sidebar.selectbox(
    "Select Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest")
)

if uploaded_file is not None:
    # Read the data
    test_df = pd.read_csv(uploaded_file)
    st.write("First 5 rows of uploaded data:")
    st.dataframe(test_df.head())

    # Separate X and y
    y_test = test_df['Diabetes_binary']
    X_test = test_df.drop('Diabetes_binary', axis=1)

    # Load the scaler and transform data
    scaler = joblib.load('model/scaler.joblib')
    X_test_scaled = scaler.transform(X_test)

    # Load the right model using if/else statements
    if model_name == "Logistic Regression":
        model = joblib.load('model/logistic_regression.joblib')
    elif model_name == "Decision Tree":
        model = joblib.load('model/decision_tree.joblib')
    elif model_name == "KNN":
        model = joblib.load('model/knn.joblib')
    elif model_name == "Naive Bayes":
        model = joblib.load('model/naive_bayes.joblib')
    elif model_name == "Random Forest":
        model = joblib.load('model/random_forest.joblib')

    st.success("Model loaded!")

    # Getting predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # c. Display of evaluation metrics
    st.subheader("Evaluation Metrics")

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Printing metrics directly to the screen
    st.write("**Accuracy:**", round(acc, 4))
    st.write("**AUC Score:**", round(auc, 4))
    st.write("**Precision:**", round(prec, 4))
    st.write("**Recall:**", round(rec, 4))
    st.write("**F1 Score:**", round(f1, 4))
    st.write("**MCC Score:**", round(mcc, 4))

    # d. Confusion matrix or classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file from the sidebar to see the results.")