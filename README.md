# 🔧 Predictive Maintenance with Sensor Analytics

## 📌 Overview

Predictive maintenance is critical in industrial environments to reduce unplanned downtime, cut costs, and enhance operational reliability. This project builds a machine learning model to predict machine failures using real-time sensor data, enabling proactive maintenance decisions.

Using features such as **Temperature**, **Vibration**, **Power Usage**, **Humidity**, and **Machine Type**, this model classifies whether a machine is likely to fail or continue operating normally. The goal is to minimize unexpected breakdowns and improve maintenance scheduling through data-driven insights.

---

## 📂 Dataset

* **Features**:

  * `Temperature (°C)`: Internal machine heat
  * `Vibration (Hz)`: Mechanical vibration frequency
  * `Power Usage (kW)`: Energy consumed by the machine
  * `Humidity (%)`: Environmental humidity
  * `Machine Type`: Categorical variable (e.g., Drill, Lathe, Mill)
  * `Failure Risk`: Binary target (0 = No failure, 1 = Failure)

---

## ⚙️ Preprocessing Steps

1. **Missing Values**: No missing values found.
2. **Categorical Encoding**: Label Encoding for `Machine Type`.
3. **Class Imbalance**: Addressed using **SMOTE** to oversample the minority class (failure instances).
4. **Feature Scaling**: MinMaxScaler applied to normalize sensor readings.
5. **Outlier Removal**: Z-score method used to eliminate extreme outliers.
6. **Correlation Check**: Verified low multicollinearity among features.

---

## 🔍 Exploratory Data Analysis

* Generated heatmaps to explore feature correlations
* Assessed distribution of target variable
* Visualized feature distributions before and after outlier removal

---

## 🧐 Model Development

### ✅ Chosen Model

* **Random Forest Classifier**
* Selected for its ability to model non-linear patterns and handle feature-rich datasets

### 🔧 Hyperparameter Tuning

Used **GridSearchCV** to optimize:

* `n_estimators`: \[100, 200, 300]
* `max_depth`: \[10, 20, None]
* `min_samples_split`: \[2, 5, 10]
* `min_samples_leaf`: \[1, 2, 4]
* `bootstrap`: \[True, False]

---

## 📈 Model Performance

| Metric        | Score |
| ------------- | ----- |
| Accuracy      | 72%   |
| Precision     | 0.69  |
| Recall        | 0.78  |
| F1-Score      | 0.73  |
| ROC-AUC Score | 0.78  |

> ✅ **High recall (78%)** ensures most failure cases are detected — crucial for early maintenance intervention.

---

## 📊 Visualizations

* **Confusion Matrix**: Visualizes true vs predicted classifications
* **ROC Curve**: Shows TPR vs FPR across thresholds, AUC = 0.78

---

## ⚠️ Challenges

* **Class Imbalance**: Required SMOTE to balance failure vs non-failure examples.
* **Overfitting Risk**: Random Forest initially overfit; mitigated via tuning and cross-validation.
* **False Positives**: Acceptable in predictive maintenance scenarios, where early warnings are preferable to missed failures.

---

## 💡 Key Learnings

* Ensemble models like Random Forest are effective in failure detection problems.
* Proper preprocessing (scaling, balancing, encoding) significantly enhances model reliability.
* High recall is more valuable than high precision in maintenance-critical applications.

---

## 🔭 Future Enhancements

* Try advanced models: **XGBoost**, **LightGBM**, or **Neural Networks**
* Use **RandomizedSearchCV** for faster tuning on large parameter grids
* Add new sensor features or external contextual data
* Set up a **model retraining pipeline** to handle drift and keep predictions relevant
* Deploy the model using Flask/FastAPI for real-time prediction

---

## 🚀 Getting Started

### Requirements

* Python 3.8+
* pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook Machine_Failure_Prediction.ipynb
```



