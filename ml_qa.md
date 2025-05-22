## Table of Contents

- [Can you explain the Evaluation Metrics for classification metrics and regression?](#1)
- [Why do we need normalization? What are some normalization methods?](#2)
- [What are Local Minima and Global Minimum?](#3)
- [What is the curse of dimensionality?](#4)
- [What are common dimensionality reduction techniques?](#5)
- [Why is PCA considered a dimensionality reduction, compression, and denoising technique?](#6)
- [What do LSA, LDA, and SVD stand for and how are they related?](#7)
- [How would you explain a Markov Chain to a high school student?](#8)
- [How would you extract topics from a text corpus?](#9)
- [Why does SVM work by expanding dimensions? What makes SVM effective?](#10)
- [Can you argue the advantages of Naive Bayes despite it being an older ML method?](#11)
- [What are appropriate metrics for regression/classification tasks?](#12)
- [Explain Support, Confidence, and Lift in Association Rule Learning.](#13)
- [Do you know about Newton’s Method and Gradient Descent?](#14)
- [What are your thoughts on the difference between machine learning and statistical approaches?](#15)
- [What were the typical problems with traditional (pre-deep-learning) neural networks?](#16)
- [What do you think is the foundation of current innovations in deep learning?](#17)
- [Can you explain the ROC curve?](#18)
- [You have 100 servers. Why might you choose Random Forest over a neural network?](#19)
- [What are K-means’ semantic limitations (aside from computation)?](#20)
- [Explain L1 and L2 regularization.](#21)
- [What is Cross Validation and how is it done?](#22)
- [Are you familiar with XGBoost? Why is it so popular in Kaggle competitions?](#23)
- [What are some ensemble methods?](#24)
- [What is a feature vector?](#25)
- [How do you define a "good" model?](#26)
- [Are 50 small decision trees better than one big tree? Why?](#27)
- [Why is logistic regression commonly used in spam filters?](#28)
- [What is the formula for OLS (Ordinary Least Squares) regression?](#29)

---

## #1

### Can you explain the metrics you know? (e.g., RMSE, MAE, recall, precision...)

Evaluation metrics can broadly be categorized into **classification metrics** and **regression metrics**.

---

### **Classification Metrics**

> **Accuracy**

Accuracy measures **how often the model makes correct predictions**. It is calculated as:

```
Accuracy = (Number of correct predictions) / (Total number of predictions)
```

Accuracy is **not reliable** when the data is imbalanced. For instance, if 90% of labels are 0 and the model always predicts 0, it still gets 90% accuracy — misleadingly high.

> **Confusion Matrix**

![Confusion Matrix](./img/2-machine-learning/confusion-matrix.png)

A confusion matrix shows **how often predictions are confused**. Common in binary classification, it consists of:
- **TP (True Positive)**: Predicted positive, actually positive.
- **FP (False Positive)**: Predicted positive, actually negative.
- **FN (False Negative)**: Predicted negative, actually positive.
- **TN (True Negative)**: Predicted negative, actually negative.

From this, accuracy can be calculated as:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

> **Precision and Recall**

Precision and Recall focus on **performance regarding positive class predictions**:

- **Precision** = TP / (TP + FP): Of all positive predictions, how many were correct?
- **Recall** = TP / (TP + FN): Of all actual positives, how many were correctly predicted?

They **trade off**: increasing precision can reduce recall and vice versa. A good model balances both.

> **F1 Score**

F1 Score is the **harmonic mean of precision and recall**, representing a balanced metric.

```
F1 = 2 * (precision * recall) / (precision + recall)
```

> **ROC-AUC**

![ROC Curve](./img/2-machine-learning/roc-curve.png)

ROC (Receiver Operating Characteristic) shows how **TPR (Recall)** varies with **FPR (False Positive Rate)** as the threshold changes.

- **TPR (Recall)** = TP / (TP + FN)
- **FPR** = FP / (FP + TN)

AUC (Area Under the Curve) reflects the **overall model performance**. Higher AUC means the ROC curve is skewed toward the top-left, indicating better performance.

---

### **Regression Metrics**

> **MAE (Mean Absolute Error)**

The **average of absolute errors** between predicted and actual values.

```
MAE = (1/N) * Σ |y_i - ŷ_i|
```

> **MSE (Mean Squared Error)**

The **average of squared errors** — more sensitive to outliers due to squaring.

```
MSE = (1/N) * Σ (y_i - ŷ_i)^2
```

> **RMSE (Root Mean Squared Error)**

The square root of MSE — in the same unit as the target.

```
RMSE = √MSE
```

> **RMSLE (Root Mean Squared Logarithmic Error)**

Like RMSE but on the **logarithmic scale**, useful when the target has exponential growth:

```
RMSLE = √(1/N * Σ (log(y_i + 1) - log(ŷ_i + 1))^2)
```

> **R-squared (Coefficient of Determination)**

Measures how much **variance in the target variable** is explained by the model. Ranges from 0 to 1.

---
