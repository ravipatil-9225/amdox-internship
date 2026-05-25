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

### Algorithms Used
* **K-Means:** Primary clustering with silhouette-optimized cluster count (6 segments).
* **DBSCAN:** Anomaly/outlier detection — identifies customers with unusual purchasing patterns.
* **Gaussian Mixture Model (GMM):** Probabilistic soft assignments for segment overlap analysis.

---

## 4. Stacked Ensemble Churn Model (XGBoost + LightGBM)

### Model Details
* **Developed by:** NeuralRetail Data Science Team (April 2026)
* **Model Type:** Stacked Ensemble — XGBoost + LightGBM DART base learners with validation-AUC-weighted meta-learner
* **Version:** 1.0.0
* **License:** Internal Use Only (Amdox Technologies)

### Intended Use
* **Primary Use Case:** Production churn prediction with higher accuracy than single-model alternatives, used as the challenger to the standalone XGBoost champion model.
* **Intended Users:** CRM Teams, Data Scientists (model comparison).

### Training Data
* **Source:** Same as XGBoost Churn model — aggregated RFM features from `data/bronze/`.
* **Features:** Age, Recency, Frequency, Monetary value.
* **Method:** 5-fold stratified cross-validation with out-of-fold prediction stacking.

### Evaluation
* **Ensemble Strategy:** Auto-selected between validation-weighted average and Logistic Regression meta-learner.
* **Current Performance:** AUC-ROC = 0.99 (on OOF predictions), F1 = 0.92 at threshold 0.35.

### Bias, Fairness, and Limitations
* **Limitations:** Slightly higher inference latency than the single XGBoost model (~2x). Requires both base models to be loaded at prediction time.

---

## 5. Price Elasticity Model (DoWhy + EconML)

### Model Details
* **Developed by:** NeuralRetail Data Science Team (April 2026)
* **Model Type:** Causal Inference — LinearDML + NonParamDML
* **Version:** 1.0.0
* **License:** Internal Use Only (Amdox Technologies)

### Intended Use
* **Primary Use Case:** Estimating causal price-demand elasticity per product category. Inputs to the what-if revenue simulator and pricing strategy recommendations.
* **Intended Users:** Finance Controllers, Category Managers.
* **Out of Scope:** Real-time dynamic pricing. Elasticity coefficients are batch-computed.

### Training Data
* **Source:** `data/bronze/transactions.parquet` + `products.parquet`.
* **Features:** Log-price (treatment), log-demand (outcome), competitor price (confounder), promotion flag (effect modifier).

### Evaluation
* **Primary Metric:** R² of log-linear model and confidence intervals on causal estimands.
* **Target:** R² ≥ 0.72.
* **Methods:** OLS baseline, DoWhy LinearDML (causal), NonParamDML with GBR (non-linear curves), Cross-price elasticity via Double ML.

### Bias, Fairness, and Limitations
* **Limitations:** Competitor pricing is synthetically generated from own-price distributions. In production, requires real competitor data feeds. Causal identifiability depends on the confounder set being sufficient.

---

## 6. Inventory EOQ Model (Analytical)

### Model Details
* **Developed by:** NeuralRetail Data Science Team (April 2026)
* **Model Type:** Analytical — Economic Order Quantity formula + Safety Stock calculation
* **Version:** 1.0.0

### Intended Use
* **Primary Use Case:** Automated purchase order recommendations, dead-stock flagging, and ABC-XYZ classification for inventory management.
* **Intended Users:** Supply Chain Analysts.

### Parameters
* **Ordering Cost:** $50/order (configurable).
* **Holding Cost Rate:** 20% of unit cost per year.
* **Service Level:** 95% (z-score = 1.65).
* **Lead Time:** 7 days (configurable).

### Evaluation
* **Primary Metric:** Stockout rate reduction.
* **Target:** Stockout ↓ 30%.
* **Method:** Comparison of EOQ-recommended reorder quantities vs. historical stockout events.

---

## 7. Revenue Forecasting Model (LightGBM + Prophet Hybrid)

### Model Details
* **Developed by:** NeuralRetail Data Science Team (April 2026)
* **Model Type:** Hybrid — LightGBM for feature-driven prediction + Prophet for trend/seasonality decomposition
* **Version:** 1.0.0

### Intended Use
* **Primary Use Case:** Monthly and quarterly revenue projections for financial planning and budget allocation.
* **Intended Users:** Finance Controllers, C-Suite Executives.

### Evaluation
* **Primary Metric:** MAPE and directional accuracy.
* **Target:** MAPE ≤ 8%.
* **Method:** LightGBM captures feature interactions (promotions, holidays, category mix). Prophet captures long-term trends and seasonality. Weighted average of both predictions.
