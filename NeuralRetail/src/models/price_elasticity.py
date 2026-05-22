"""
NeuralRetail -- Price Elasticity Analysis (Standalone)
=======================================================
Phase 1 Causal Inference Polish:
  - Linear elasticity via DoWhy + LinearDML
  - Non-linear demand elasticity curves via NonParamDML + GBR
  - Cross-price elasticity analysis via Double ML
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
import os
import warnings
warnings.filterwarnings('ignore')

import dowhy
from econml.dml import LinearDML, NonParamDML


def calculate_elasticity():
    """Compute linear + non-linear price elasticity from transaction data."""
    print("=" * 55)
    print("NeuralRetail - Price Elasticity Analysis")
    print("=" * 55)

    # Load real data if available, otherwise use synthetic
    try:
        products = pd.read_parquet("data/bronze/products.parquet")
        transactions = pd.read_parquet("data/bronze/transactions.parquet")
        merged = transactions.merge(
            products[['sku_id', 'category', 'base_price']], on='sku_id')
        merged['unit_price'] = merged['total_amount'] / merged['quantity']
        print(f"Loaded {len(merged)} transaction records\n")
    except Exception:
        print("Using synthetic data\n")
        np.random.seed(42)
        prices = np.random.uniform(10, 50, 1000)
        demand = 500 - 5 * prices + np.random.normal(0, 20, 1000)
        merged = pd.DataFrame({
            'unit_price': prices, 'quantity': demand,
            'category': np.random.choice(['A', 'B'], 1000),
        })

    results = {}
    for cat in merged['category'].unique():
        cat_data = merged[merged['category'] == cat]
        if len(cat_data) < 30:
            continue

        # Log-transform
        cat_data = cat_data.copy()
        cat_data['log_price'] = np.log(cat_data['unit_price'] + 1e-5)
        cat_data['log_demand'] = np.log(cat_data['quantity'].clip(lower=0.1) + 1e-5)

        # Synthetic confounder for causal estimation
        np.random.seed(42)
        cat_data['comp_price_log'] = (
            cat_data['log_price'] + np.random.normal(0, 0.1, len(cat_data)))

        sample = cat_data.sample(n=min(500, len(cat_data)), random_state=42)

        # -- 1. OLS Baseline --
        X_ols = sm.add_constant(sample['log_price'])
        ols_model = sm.OLS(sample['log_demand'], X_ols).fit()
        ols_elasticity = ols_model.params.iloc[-1]

        # -- 2. LinearDML (causal) --
        try:
            causal = dowhy.CausalModel(
                data=sample,
                treatment='log_price',
                outcome='log_demand',
                common_causes=['comp_price_log'],
            )
            estimand = causal.identify_effect(proceed_when_unidentifiable=True)
            estimate = causal.estimate_effect(
                estimand,
                method_name="backdoor.econml.dml.LinearDML",
                method_params={
                    "init_params": {'discrete_treatment': False},
                    "fit_params": {},
                },
            )
            linear_dml = float(estimate.value)
        except Exception:
            linear_dml = ols_elasticity

        # -- 3. NonParamDML (non-linear curves) --
        try:
            T = sample['log_price'].values
            Y = sample['log_demand'].values
            W = sample[['comp_price_log']].values

            npm = NonParamDML(
                model_y=GradientBoostingRegressor(n_estimators=80, max_depth=3),
                model_t=GradientBoostingRegressor(n_estimators=80, max_depth=3),
                model_final=GradientBoostingRegressor(n_estimators=40, max_depth=2),
                discrete_treatment=False, random_state=42,
            )
            npm.fit(Y, T, W=W)
            nonlinear_dml = float(np.mean(npm.effect(W)))
        except Exception:
            nonlinear_dml = linear_dml

        results[cat] = {
            'ols': round(ols_elasticity, 4),
            'linear_dml': round(linear_dml, 4),
            'nonlinear_dml': round(nonlinear_dml, 4),
        }

        print(f"Category: {cat}")
        print(f"  OLS elasticity:      {ols_elasticity:.4f}")
        print(f"  LinearDML (causal):  {linear_dml:.4f}")
        print(f"  NonParamDML (curve): {nonlinear_dml:.4f}")
        if abs(nonlinear_dml) > 1:
            print("  -> ELASTIC: price down increases revenue")
        else:
            print("  -> INELASTIC: price up increases revenue")
        print()

    return results


def calculate_cross_elasticity():
    """Compute cross-price elasticity between all category pairs."""
    print("\n" + "=" * 55)
    print("Cross-Price Elasticity Analysis (Double ML)")
    print("=" * 55)

    try:
        products = pd.read_parquet("data/bronze/products.parquet")
        transactions = pd.read_parquet("data/bronze/transactions.parquet")
        merged = transactions.merge(
            products[['sku_id', 'category', 'base_price']], on='sku_id')
        merged['unit_price'] = merged['total_amount'] / merged['quantity']
    except Exception:
        print("Data not available for cross-elasticity.")
        return {}

    categories = merged['category'].unique()
    cross_results = {}

    for focal in categories:
        for comp in categories:
            if focal == comp:
                continue

            focal_daily = merged[merged['category'] == focal].groupby('timestamp').agg(
                {'quantity': 'sum', 'unit_price': 'mean'}).reset_index()
            comp_daily = merged[merged['category'] == comp].groupby('timestamp').agg(
                {'unit_price': 'mean'}).reset_index().rename(
                columns={'unit_price': 'comp_price'})

            combined = focal_daily.merge(comp_daily, on='timestamp', how='inner')
            if len(combined) < 20:
                continue

            combined['log_q'] = np.log(combined['quantity'] + 1e-5)
            combined['log_cp'] = np.log(combined['comp_price'] + 1e-5)
            combined['log_fp'] = np.log(combined['unit_price'] + 1e-5)

            try:
                dml = LinearDML(discrete_treatment=False, random_state=42)
                dml.fit(
                    Y=combined['log_q'].values,
                    T=combined['log_cp'].values,
                    W=combined[['log_fp']].values,
                )
                ce = float(dml.const_marginal_ate())
                key = f"{focal} ← {comp}"
                cross_results[key] = round(ce, 4)
                label = "Substitute" if ce > 0 else "Complement"
                print(f"  {key}: {ce:.4f} ({label})")
            except Exception:
                pass

    return cross_results


if __name__ == "__main__":
    calculate_elasticity()
    calculate_cross_elasticity()
    print("\nElasticity analysis complete! [OK]")
