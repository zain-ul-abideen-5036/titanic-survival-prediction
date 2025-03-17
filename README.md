# Titanic Survival Prediction with Naive Bayes

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)

A machine learning project demonstrating Naive Bayes classification for predicting Titanic survival outcomes, featuring data preprocessing, model evaluation, and visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#Contributing)
- [Contact](#Contact)

---

## Project Overview
This project demonstrates:
- Data preprocessing for real-world datasets
- Implementation of Gaussian Naive Bayes classifier
- Model evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Visualization techniques (Confusion Matrix, ROC Curve, Feature Importance)
- Handling of missing values and categorical data

---

## Dataset
The Titanic dataset contains information about 887 passengers including:
- Survival status (0 = No, 1 = Yes)
- Passenger class (1st, 2nd, 3rd)
- Sex
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Fare paid

Source: [Stanford Titanic Dataset](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv)

---

## Features
- **Input Features**:
  - Pclass (Passenger class)
  - Sex
  - Age
  - Siblings/Spouses Aboard
  - Parents/Children Aboard
  - Fare
  
- **Target Variable**:
  - Survived (Binary classification)

## Methodology
1. **Data Preprocessing**:
   - Missing value imputation (Median for Age)
   - Categorical encoding (Sex to binary)
   - Feature scaling (StandardScaler)

2. **Model Training**:
   - Gaussian Naive Bayes classifier
   - 70-30 train-test split
   - Random state fixed for reproducibility

3. **Evaluation**:
   - Accuracy score
   - Classification report
   - Confusion matrix
   - ROC-AUC curve

4. **Visualization**:
   - Decision boundaries analysis
   - Feature importance visualization
   - Model performance metrics

---

## Results
### Model Performance
```text
Accuracy: 0.7640449438202247

Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.86      0.82       166
           1       0.73      0.60      0.66       101

    accuracy                           0.76       267
   macro avg       0.75      0.73      0.74       267
weighted avg       0.76      0.76      0.76       267
```

---

## Visualizations
1. Confusion Matrix
   
![image](https://github.com/user-attachments/assets/6ccd942a-3d93-4169-afd4-23544f4a8bf0)
 
2. ROC Curve
   
![image](https://github.com/user-attachments/assets/64be7a18-238a-48f3-a6e7-a08c9f70ed35)


3. Feature Importance
   
![image](https://github.com/user-attachments/assets/79d133d9-f6a2-438c-b35f-a93105dcaffa)

---

## Installation
1. Clone the repository:
   ```bash
      git clone https://github.com/zain-ul-abideen-5036/titanic-survival-prediction.git
      cd titanic-survival-prediction
   ```
---

## Usage
1. Clone the repository:
   ```bash
    jupyter notebook titanic_naive_bayes.ipynb
   ```
2. Main components:
   - Data loading and exploration
   - Data preprocessing pipeline
   - Model training and evaluation
   - Visualization generation
---

## Contributing
- Fork the repository
- Create feature branch
- Submit PR with detailed description
---

## Contact
For questions or feedback, contact: abideen5036@gmail.com

---
