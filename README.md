# CreditRisk Analytics
This repository contains a machine learning web application built with Streamlit for predicting credit risk and borrower default using trained classification models (Logistic Regression, Decision Tree, and Gradient Boosting). The application provides an interactive user interface to input financial and personal features to instantly evaluate creditworthiness and default risk.

EDA & Preprocessing: Loaded the Credit Risk dataset, handled missing values (median imputation), and analyzed distributions and statistics using Pandas, NumPy, Matplotlib, and Seaborn.

Feature Engineering: Processed features, converted categorical variables, handled class imbalance and normalization, and split the dataset using train_test_split.

Models: Logistic Regression, Decision Tree, and Gradient Boosting.

Classification Threshold: Applied classification thresholds to optimize decision boundaries for risk prediction.

Model Interpretability: Utilized SHAP and LIME to interpret model predictions and explain individual feature impacts ethically and transparently.
you can see app there : https://creditrisk-analytics.streamlit.app/
