# NeuralRetail Model Cards

## 1. Demand Forecasting Model (Prophet + LSTM Ensemble)

### Model Details
* **Developed by:** NeuralRetail Data Science Team (April 2026)
* **Model Type:** Hybrid Statistical & Deep Learning Time-Series Ensemble
* **Architecture:** Weighted average of Facebook Prophet (additive seasonality) and PyTorch Lightning LSTM (28-day lookback window).
* **Version:** 1.2.0
* **License:** Internal Use Only (Amdox Technologies)

### Intended Use
* **Primary Use Case:** SKU-level demand forecasting at daily, weekly, and monthly granularity to inform inventory replenishment and capacity planning.
* **Intended Users:** Supply Chain Analysts, Category Managers.
* **Out of Scope:** Intra-day (hourly) forecasting, forecasting for entirely new SKUs without historical data (cold-start problem).

### Training Data
* **Source:** Historical POS transactions from the `data/bronze/transactions.parquet` dataset.
* **Timeframe:** 2 years of daily aggregated sales data.
* **Features:** Target (sales volume), lag features (t-1, t-7, t-14), date features (day-of-week, holiday flags), and external regressors (promotional events).

### Evaluation
* **Primary Metric:** Mean Absolute Percentage Error (MAPE).
* **Target:** ≤ 10% on a 30-day forecast horizon.
* **Current Performance:** MAPE = 8.5% (Validation Set), 90% Prediction Interval Coverage = 89.2%.

### Bias, Fairness, and Limitations
* **Bias Evaluation:** Forecasts were evaluated across different product categories (e.g., Electronics vs. Groceries). Performance is slightly lower (MAPE ~12%) for highly volatile, low-volume "long tail" SKUs.
* **Limitations:** The model assumes that future promotional impact will resemble historical promotional impact. Unprecedented macroeconomic shocks (e.g., supply chain crisis) are not automatically accounted for without manual intervention.

---

## 2. Customer Churn Prediction Model (XGBoost)

### Model Details
* **Developed by:** NeuralRetail Data Science Team (April 2026)
* **Model Type:** Gradient Boosted Decision Tree (XGBoost Classifier)
* **Version:** 2.0.1
* **License:** Internal Use Only (Amdox Technologies)

### Intended Use
* **Primary Use Case:** Predicting the probability (0-1) that a customer will churn (no purchase in the next 30 days) to trigger targeted retention campaigns.
* **Intended Users:** CRM and Marketing Teams.
* **Out of Scope:** Predicting lifetime value (CLV) directly, or predicting churn for guest users without a registered profile.

### Training Data
* **Source:** Aggregated customer profiles and RFM (Recency, Frequency, Monetary) metrics from `data/bronze/customers.parquet` and `transactions.parquet`.
* **Features:** Age, Recency, Frequency, Monetary value, Category affinity.
* **Class Imbalance:** Churners typically represent ~20% of the dataset. Handled using `scale_pos_weight` in XGBoost.

### Evaluation
* **Primary Metric:** Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
* **Target:** AUC-ROC ≥ 0.90.
* **Current Performance:** AUC-ROC = 0.99 (Champion Model), Precision@Top20% = 0.85.

### Bias, Fairness, and Limitations
* **Explainability:** SHAP (SHapley Additive exPlanations) is used to generate feature importance and local explanations for every prediction.
* **Bias Evaluation:** Model was audited for disparate impact across age demographics. The false positive rate variance between Age < 30 and Age > 50 cohorts is within the acceptable 5% threshold.
* **Limitations:** Highly reliant on the "Recency" feature. Customers who make large, infrequent purchases (e.g., luxury goods) may be falsely flagged as high churn risk if their natural purchase cycle exceeds 90 days.

---

## 3. Customer Segmentation Model (K-Means)

### Model Details
* **Developed by:** NeuralRetail Data Science Team (April 2026)
* **Model Type:** Unsupervised Clustering (K-Means)
* **Version:** 1.0.0
* **License:** Internal Use Only (Amdox Technologies)

### Intended Use
* **Primary Use Case:** Grouping customers into distinct behavioural personas (e.g., "Champions", "At Risk", "New Spenders") based on purchasing habits.
* **Intended Users:** Marketing and Executive Teams.

### Evaluation
* **Primary Metric:** Silhouette Score.
* **Current Performance:** Silhouette Score = 0.62 (Target ≥ 0.55). Stable across 6 distinct clusters.
