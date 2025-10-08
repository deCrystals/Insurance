# train.py
"""
Train multiple models for Ultimate_Claim_Amount, pick the best by RMSE (on £ scale),
save the best full pipeline, and compute **global SHAP feature importances**.
Outputs:
- best_claim_model.joblib            (full sklearn Pipeline)
- training_report.json               (metrics + best model + SHAP top features)
- shap_feature_importance.csv        (feature, mean_abs_shap on log1p scale)
- shap_bar.png                       (bar chart of top SHAP features)
- shap_beeswarm.png                  (beeswarm summary plot of SHAP values)
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

DEFAULT_OUTPUT = "best_claim_model.joblib"
DEFAULT_REPORT = "training_report.json"

def build_preprocessor(num_features, cat_features):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocess = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])
    return preprocess

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def safe_feature_names(prep, n_cols):
    try:
        return prep.get_feature_names_out().tolist()
    except Exception:
        return [f"f{i}" for i in range(n_cols)]

def shap_explain_global(best_pipe, X_sample, out_csv="shap_feature_importance.csv",
                        out_bar="shap_bar.png", out_beeswarm="shap_beeswarm.png"):
    """Compute global SHAP (mean |SHAP|) on a sample and save CSV and plots.
    Note: SHAP values are on the **log1p** scale because the model predicts log1p(target).
    """
    import shap  # ensure shap is installed
    prep = best_pipe.named_steps["prep"]
    model = best_pipe.named_steps["model"]

    # Transform the sample with the fitted preprocessor
    X_trans = prep.transform(X_sample)
    feat_names = safe_feature_names(prep, X_trans.shape[1])
    # Build explainer that works across models
    try:
        explainer = shap.Explainer(model, X_trans)
    except Exception:
        # Fallback for tree models
        explainer = shap.TreeExplainer(model)

    sv = explainer(X_trans)  # shap.Explanation
    # Ensure feature names are present
    try:
        sv.feature_names = feat_names
    except Exception:
        pass

    # Global importance = mean absolute SHAP value per feature
    mean_abs = np.mean(np.abs(sv.values), axis=0)
    imp = (pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
             .sort_values("mean_abs_shap", ascending=False)
             .reset_index(drop=True))
    imp.to_csv(out_csv, index=False)

    # Bar chart (top 25)
    top = imp.head(25)
    plt.figure(figsize=(8, 10))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value| (log1p scale)")
    plt.title("Global Feature Importance (SHAP) – Top 25")
    plt.tight_layout()
    plt.savefig(out_bar, dpi=180)
    plt.close()

    # Beeswarm (uses SHAP's own plotting)
    try:
        shap.summary_plot(sv, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(out_beeswarm, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception:
        # If summary_plot fails (rare), we silently skip
        pass

    return imp

def main(csv_path: str,
         model_out: str = DEFAULT_OUTPUT,
         report_out: str = DEFAULT_REPORT,
         test_size: float = 0.2,
         random_state: int = 42,
         shap_sample_size: int = 1000):
    # Load data
    df = pd.read_csv(csv_path)
    df = df[df["Ultimate_Claim_Amount"].notna()].copy()

    # Dates and engineered lag
    df["Accident_Date"] = pd.to_datetime(df["Accident_Date"], dayfirst=True, errors="coerce")
    df["FNOL_Date"] = pd.to_datetime(df["FNOL_Date"], dayfirst=True, errors="coerce")
    df["Reporting_Lag"] = (df["FNOL_Date"] - df["Accident_Date"]).dt.days.clip(lower=0)

    # Features
    num_features = ["Age_of_Driver", "Annual_Mileage", "Driving_Experience_Years", "Vehicle_Age", "Reporting_Lag"]
    cat_features = [
        "Claim_Type", "Claim_Complexity", "Fraud_Flag", "Litigation_Flag", "Severity_Band",
        "Gender", "Occupation", "Region", "Vehicle_Type", "Credit_Score_Band"
    ]

    # Inputs/target
    X = df[num_features + cat_features]
    y = np.log1p(df["Ultimate_Claim_Amount"].values)  # log-transform target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Preprocessor
    preprocess = build_preprocessor(num_features, cat_features)

    # Candidate models
    models = {
        "ols": LinearRegression(),
        "ridge": Ridge(alpha=300.0),
        "rf": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1),
        "xgb": XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1, tree_method="hist"
        ),
    }

    results = {}
    best_name, best_rmse, best_pipe = None, np.inf, None

    for name, est in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", est)])
        pipe.fit(X_train, y_train)

        # Back-transform predictions to £
        y_pred = np.expm1(pipe.predict(X_test))
        y_true = df.loc[X_test.index, "Ultimate_Claim_Amount"].to_numpy(dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_t, y_p = y_true[mask], y_pred[mask]

        metrics = compute_metrics(y_t, y_p)
        results[name] = metrics

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_name = name
            best_pipe = pipe

    if best_pipe is None:
        raise RuntimeError("No best model found.")

    # Save best pipeline
    joblib.dump(best_pipe, model_out)

    # SHAP on a training sample
    n_sample = min(shap_sample_size, len(X_train))
    X_sample = X_train.sample(n=n_sample, random_state=random_state)
    shap_imp = shap_explain_global(best_pipe, X_sample)

    # Save report
    report = {
        "best_model": best_name,
        "metrics": results,
        "model_path": model_out,
        "features": {
            "numeric": num_features,
            "categorical": cat_features
        },
        "shap_top_features": shap_imp.head(20).to_dict(orient="records"),
        "notes": "Metrics on original £ scale. SHAP values on log1p(target) scale."
    }
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    summary_df = pd.DataFrame(results).T.sort_values("RMSE")
    print("Summary (original £ units):")
    print(summary_df)
    print(f" Best model: {best_name}  (saved to {model_out})")
    print(f"SHAP: saved 'shap_feature_importance.csv', 'shap_bar.png', and 'shap_beeswarm.png'")
    print(f"Training report written to {report_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="policy.csv", help="Path to policy.csv")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT, help="Where to save the best pipeline")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT, help="Where to save the JSON report")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for the train/test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shap_sample_size", type=int, default=1000, help="Rows from training set for SHAP (to keep compute reasonable)")
    args = parser.parse_args()
    main(args.csv, args.out, args.report, args.test_size, args.seed, args.shap_sample_size)
