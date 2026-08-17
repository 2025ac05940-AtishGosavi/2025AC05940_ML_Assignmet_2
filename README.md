# Machine Learning Assignment 2: Classification Models & Deployment

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy multiple machine learning classification models to predict the health status of individuals based on health indicators. By developing an interactive Streamlit web application, the project demonstrates an end-to-end ML deployment workflow, allowing users to upload test data, select different predictive models, and instantly view evaluation metrics and confusion matrices.

## b. Dataset Description
- **Source:** Kaggle (Diabetes Health Indicators Dataset)
- **Problem Type:** Binary Classification
- **Description:** This dataset contains healthcare statistics and lifestyle indicators (such as BMI, high blood pressure, high cholesterol, smoker status, and physical activity) to predict whether an individual has diabetes or is at risk of diabetes (`Diabetes_binary` target variable).
- **Instances:** 20,000 (Sampled from the original dataset to ensure optimal deployment performance while exceeding the 500 minimum requirement).
- **Features:** 21 features (exceeding the 12 minimum feature requirement).

## c. Github Repository Link
**Link:** `[INSERT YOUR GITHUB REPOSITORY LINK HERE]`

## d. Models Used and Evaluation Metrics
The following 5 classification models were implemented and evaluated on the test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8685 | 0.8409 | 0.5033 | 0.1442 | 0.2242 | 0.2176 |
| Decision Tree | 0.820  | 0.6024 | 0.2838 | 0.3302 | 0.3053 | 0.1913 |
| kNN | 0.8605 | 0.7388 | 0.4362 | 0.2011 | 0.2753 | 0.2289 |
| Naive Bayes | 0.7843 | 0.8054 | 0.3268 | 0.6015 | 0.4235 | 0.3263 |
| Random Forest (Ensemble) | 0.8682 | 0.8177 | 0.5    | 0.1252 | 0.2003 | 0.2011 |


## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved the highest overall Accuracy (86.85%) and the best AUC score (0.8409). However, it struggled significantly with Recall (0.1442), meaning it missed many positive diabetes cases. |
| Decision Tree | Showed the lowest AUC (0.6024) and MCC (0.1913), indicating that a single unpruned tree struggled to generalize well on this dataset compared to the other algorithms. |
| kNN | Delivered strong Accuracy (86.05%) but average performance in capturing the minority class, resulting in relatively low Recall and F1 scores. |
| Naive Bayes | Despite having the lowest Accuracy (78.43%), it performed exceptionally well at identifying positive cases. It achieved the highest Recall (0.6015), F1 Score (0.4235), and MCC (0.3263) of all models tested. |
| Random Forest (Ensemble) | Matched Logistic Regression in high Accuracy (86.82%) and precision but suffered from a very low Recall (0.1252), suggesting the ensemble favored predicting the majority class. |
| **Overall Winner for your dataset?** | **Naive Bayes** (For medical diagnosis, capturing true positive cases is critical. Naive Bayes drastically outperformed the others in Recall, F1, and MCC, making it the most practical model despite lower overall accuracy). |