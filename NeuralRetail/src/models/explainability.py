"""
NeuralRetail -- Unified Explainability Module
===============================================
Provides three explainability backends:
  1. SHAP   - For tree-based models (XGBoost, LightGBM)
  2. LIME   - Fallback for any scikit-learn-compatible model
  3. Captum - For PyTorch models (LSTM forecaster)

Each backend returns a dict of {feature_name: importance_score} for
consistent downstream consumption by the API and dashboard.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("neuralretail.explain")


# ---------------------------------------------------------------------------
# 1. SHAP Explainer (Primary for tree models)
# ---------------------------------------------------------------------------

def explain_shap(model, X: pd.DataFrame, instance_idx: int = 0) -> dict:
    """
    Generate SHAP values for a single instance using TreeExplainer.
    Falls back to KernelExplainer if TreeExplainer is unavailable.

    Args:
        model: Trained tree model (XGBoost, LightGBM, sklearn).
        X: Feature DataFrame.
        instance_idx: Index of instance to explain.

    Returns:
        Dict of {feature_name: shap_value}.
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fallback for non-tree models
        background = shap.sample(X, min(50, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X.iloc[[instance_idx]])

    # Handle multi-output (binary classification returns list of 2 arrays)
    if isinstance(shap_values, list):
        vals = shap_values[1][0]  # Class 1 (churn) SHAP values
    else:
        vals = shap_values[0]

    return {col: round(float(v), 6) for col, v in zip(X.columns, vals)}


def shap_summary(model, X: pd.DataFrame, max_display: int = 10) -> dict:
    """
    Generate global SHAP feature importance summary.

    Returns:
        Dict of {feature_name: mean_abs_shap_value}, sorted descending.
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        background = shap.sample(X, min(50, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = {col: round(float(v), 6) for col, v in zip(X.columns, mean_abs)}
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:max_display])


# ---------------------------------------------------------------------------
# 2. LIME Explainer (Fallback for any sklearn-compatible model)
# ---------------------------------------------------------------------------

def explain_lime(model, X: pd.DataFrame, instance_idx: int = 0,
                 num_features: int = 10) -> dict:
    """
    Generate LIME explanation for a single instance.

    Args:
        model: Any model with predict_proba method.
        X: Feature DataFrame.
        instance_idx: Index of instance to explain.
        num_features: Max features in explanation.

    Returns:
        Dict of {feature_name: lime_weight}.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        logger.warning("LIME not installed. Install with: pip install lime")
        return {}

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["No Churn", "Churn"],
        mode="classification",
        random_state=42,
    )

    instance = X.iloc[instance_idx].values
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features,
    )

    return {feat: round(float(weight), 6)
            for feat, weight in explanation.as_list()}


# ---------------------------------------------------------------------------
# 3. Captum Explainer (For PyTorch models)
# ---------------------------------------------------------------------------

def explain_captum(model, input_tensor, feature_names: list = None,
                   method: str = "integrated_gradients") -> dict:
    """
    Generate Captum attributions for a PyTorch model.

    Args:
        model: PyTorch nn.Module (e.g., LSTMForecaster).
        input_tensor: Input tensor (batch_size=1, seq_len, features).
        feature_names: Optional list of feature names.
        method: Attribution method ('integrated_gradients' or 'saliency').

    Returns:
        Dict of {feature/timestep: attribution_score}.
    """
    try:
        import torch
        from captum.attr import IntegratedGradients, Saliency
    except ImportError:
        logger.warning("Captum not installed. Install with: pip install captum")
        return {}

    model.eval()

    if method == "integrated_gradients":
        ig = IntegratedGradients(model)
        baseline = torch.zeros_like(input_tensor)
        attributions = ig.attribute(input_tensor, baselines=baseline)
    elif method == "saliency":
        sal = Saliency(model)
        attributions = sal.attribute(input_tensor)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Average across sequence dimension for time-series
    attr_values = attributions.squeeze().detach().numpy()
    if attr_values.ndim == 2:
        # (seq_len, features) -> average across timesteps
        attr_values = np.abs(attr_values).mean(axis=0)
    elif attr_values.ndim == 1:
        attr_values = np.abs(attr_values)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(attr_values))]

    return {name: round(float(v), 6)
            for name, v in zip(feature_names, attr_values)}


# ---------------------------------------------------------------------------
# Unified explainer interface
# ---------------------------------------------------------------------------

def explain(model, X, instance_idx: int = 0,
            method: str = "auto", **kwargs) -> dict:
    """
    Unified explanation interface that auto-selects the best backend.

    Args:
        model: Trained model (sklearn, XGBoost, PyTorch, etc.)
        X: Feature data (DataFrame for sklearn/tree, Tensor for PyTorch).
        instance_idx: Index to explain.
        method: 'auto', 'shap', 'lime', or 'captum'.

    Returns:
        Dict of {feature_name: importance_score}.
    """
    if method == "auto":
        # Auto-detect model type
        model_type = type(model).__name__
        if model_type in ("XGBClassifier", "XGBRegressor",
                          "LGBMClassifier", "LGBMRegressor",
                          "GradientBoostingClassifier",
                          "RandomForestClassifier"):
            method = "shap"
        elif hasattr(model, "parameters"):
            # PyTorch model
            method = "captum"
        else:
            method = "lime"

    logger.info(f"[Explain] Using {method} backend for {type(model).__name__}")

    if method == "shap":
        return explain_shap(model, X, instance_idx)
    elif method == "lime":
        return explain_lime(model, X, instance_idx, **kwargs)
    elif method == "captum":
        return explain_captum(model, X, **kwargs)
    else:
        raise ValueError(f"Unknown explainability method: {method}")
