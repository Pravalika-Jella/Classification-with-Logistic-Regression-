## Objective

The objective of this task is to implement a Logistic Regression model for binary classification using the Breast Cancer Wisconsin dataset from scikit-learn. The model predicts whether a tumor is malignant or benign based on diagnostic features.

---

## Dataset

Source: load_breast_cancer() from scikit-learn

Samples: 569

Features: 30 numeric features (e.g., mean radius, mean texture, mean smoothness, etc.)

## Target Classes:

0 → Malignant

1 → Benign

---

## Steps Performed

1. Data Loading

Loaded the breast cancer dataset and converted it into a Pandas DataFrame and Series.



2. Data Preprocessing

Train-test split with 80% training and 20% testing, using stratification to maintain class balance.

Standardized features using StandardScaler to normalize data for logistic regression.



3. Model Training

Used LogisticRegression with default hyperparameters (random_state=42).

Trained on scaled training data.



4. Model Evaluation

Predictions and probabilities generated for the test set.

Calculated performance metrics:

Accuracy

Precision

Recall

ROC-AUC

Confusion Matrix




5. Visualization

Plotted ROC Curve to visualize classification performance.

Plotted Sigmoid Function to show probability mapping.

---

Model Evaluation Metrics

Accuracy      : 0.97
Precision     : 0.97
Recall        : 0.98
ROC-AUC Score : 0.99
Confusion Matrix:
[[39  4]
 [ 0 71]]


---

## Visualizations

ROC Curve

Shows the trade-off between True Positive Rate and False Positive Rate.

AUC = 0.99, indicating excellent classification performance.


## Sigmoid Function

Demonstrates how logistic regression maps linear outputs to probabilities between 0 and 1.


---

## Dependencies

Install the required Python libraries before running the script:

pip install numpy pandas scikit-learn matplotlib

---

## How to Run

python task4_logistic_regression.py


---
