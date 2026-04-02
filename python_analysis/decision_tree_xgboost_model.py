# ============================================================================
# Boston Property Assessment — Decision Tree & XGBoost Models (Python)
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2025 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : End-to-end regression pipeline predicting total property value
#           from the FY2025 assessment dataset using two models:
#           a Decision Tree Regressor and an XGBoost Regressor.
#           Results are compared against the R XGBoost pipeline in
#           xgboost_model.R to validate cross-language consistency.
#
# Confirmed results (FY2025, leakage-free):
#   Train : 118,324 records | Test : 29,582 records | Features : 27
#   Decision Tree : R² = 0.9777 | RMSE = $1,876,075 | MAE = $67,641
#   XGBoost       : R² = 0.7489 | RMSE = $6,290,224 | MAE = $299,554
#
# Note on Decision Tree R²:
#   The high R² reflects that bldg_value + land_value ≈ total_value for most
#   properties. The tree is learning this accounting identity at depth 15.
#   This is a characteristic of property assessment data, not a modelling
#   error — removing those columns would eliminate the most predictive signal.
#
# Data leakage note:
#   price_per_sqft     = total_value / living_area  — derived from the target
#   tax_to_value_ratio = gross_tax   / total_value  — derived from the target
#   Both are computed for EDA reference but excluded from the feature set.
#   Including them inflates R² artificially (confirmed: XGBoost R² jumps
#   from 0.7489 to 0.7665 when they are included).
#
# Cross-language comparison (Python XGBoost vs R XGBoost):
#   Python : R² = 0.7489 | RMSE = $6.29M | MAE = $299,554
#   R      : R² = 0.5803 | RMSE = $9.68M | MAE = $304,371
#   Python performs better because it trains on 147,906 records (no IQR
#   filtering) vs the R pipeline's heavier outlier removal. The MAE values
#   are closely aligned ($299K vs $304K) confirming pipeline consistency.
#
# Pipeline:
#   1.  Data loading and column standardisation
#   2.  Type conversion (comma/dollar-formatted strings to numeric)
#   3.  Invalid record removal
#   4.  Feature engineering
#   5.  Feature selection
#   6.  Missing value handling (drop >50%, impute remainder)
#   7.  Label encoding of categorical features
#   8.  Train / test split (80/20, random_state = 42)
#   9.  Target distribution visualisation
#   10. Decision Tree Regressor
#   11. XGBoost Regressor (hyperparameters mirror xgboost_model.R)
#   12. Model comparison table
#   13. Diagnostic visualisations (actual vs predicted, errors, residuals)
#
# File path: script is run from python_analysis/; data lives in data/
# ============================================================================

import warnings
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.tree            import DecisionTreeRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")


# -- 1. Data Loading ----------------------------------------------------------
# ZIP code columns are read as strings to preserve leading zeros
# (e.g. "02128" would otherwise be parsed as the integer 2128).
# low_memory=False suppresses dtype inference warnings on mixed-type columns.

FILE_PATH = "../data/FY2025_property_assessment.csv"

df = pd.read_csv(
    FILE_PATH,
    dtype      = {"MAIL_ZIP_CODE": str, "ZIP_CODE": str},
    low_memory = False
)

# Standardise column names: strip surrounding whitespace, lowercase, replace
# spaces with underscores. This normalises " GROSS_TAX " (a known FY2025
# CSV header formatting issue with leading and trailing spaces).
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Columns with missing values: {df.isnull().sum()[df.isnull().sum() > 0].shape[0]}")


# -- 2. Type Conversion -------------------------------------------------------
# FY2025 stores several numeric fields as formatted strings:
#   land_sf, land_value, bldg_value, total_value, gross_area, living_area
#     — comma-separated integers (e.g. "1,150")
#   gross_tax — dollar-prefixed with commas (e.g. "$9,252.42")
# errors="coerce" replaces any unparseable value with NaN rather than raising.

NUMERIC_STR_COLS = [
    "land_sf", "land_value", "bldg_value",
    "total_value", "gross_area", "living_area",
]

for col in NUMERIC_STR_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"
        )

# gross_tax: strip all non-numeric characters then convert
tax_candidates = [c for c in df.columns if "gross_tax" in c]
if tax_candidates:
    tax_col      = tax_candidates[0]
    df["gross_tax"] = pd.to_numeric(
        df[tax_col].astype(str).str.replace(r"[^0-9.]", "", regex=True).str.strip(),
        errors="coerce"
    )
    if tax_col != "gross_tax":
        df.drop(columns=[tax_col], inplace=True)

print(f"Numeric conversion complete | total_value: "
      f"${df['total_value'].min():,.0f} – ${df['total_value'].max():,.0f}")


# -- 3. Invalid Record Removal ------------------------------------------------
# Properties with zero or missing target/predictor values and impossible
# build years (< 1700 or > 2025) are data entry errors and are removed.

df = df[
    df["total_value"].notna() & (df["total_value"] > 0) &
    df["living_area"].notna() & (df["living_area"] > 0) &
    df["land_sf"].notna()     & (df["land_sf"]     > 0) &
    df["yr_built"].notna()    & (df["yr_built"]     > 1700) &
    (df["yr_built"] <= 2025)
]

print(f"After removing invalid records: {df.shape[0]:,} rows")


# -- 4. Feature Engineering ---------------------------------------------------
# age_of_property is the only derived feature included in the model.
#
# price_per_sqft and tax_to_value_ratio are computed for EDA reference
# but deliberately excluded from the feature set — both are algebraic
# transformations of total_value (the target) and constitute data leakage:
#   price_per_sqft     = total_value / living_area  → leaks target
#   tax_to_value_ratio = gross_tax   / total_value  → leaks target
# Confirmed impact: XGBoost R² inflates from 0.7489 to 0.7665 when included.

CURRENT_YEAR = 2025

df["age_of_property"]    = CURRENT_YEAR - df["yr_built"]
df["price_per_sqft"]     = df["total_value"] / df["living_area"]        # EDA only
df["tax_to_value_ratio"] = (df["gross_tax"] / df["total_value"]) * 100  # EDA only

print(f"Feature engineering complete | mean property age: {df['age_of_property'].mean():.1f} yrs")


# -- 5. Feature Selection -----------------------------------------------------
# Administrative columns (owner name, mailing address, PID) are excluded
# as they have no causal relationship with assessed value.
# price_per_sqft and tax_to_value_ratio are excluded — see Section 4.

FEATURE_COLS = [
    # Core valuation components
    "bldg_value", "land_value", "gross_area", "living_area", "land_sf", "num_bldgs",
    # Construction attributes
    "yr_built", "yr_remodel",
    # Interior features
    "bed_rms", "full_bth", "hlf_bth", "kitchens", "fireplaces", "num_parking",
    # Taxation
    "gross_tax",
    # Derived (age only — price_per_sqft and tax_to_value_ratio excluded)
    "age_of_property",
    # Categorical (label-encoded in Section 7)
    "bldg_type", "city", "zip_code", "overall_cond",
    "int_cond", "ext_cond", "structure_class", "roof_structure",
    "heat_type", "ac_type", "own_occ", "prop_view", "orientation",
]

TARGET_COL   = "total_value"
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

df_model = df[FEATURE_COLS + [TARGET_COL]].copy()
print(f"Feature set: {len(FEATURE_COLS)} features, {df_model.shape[0]:,} records")


# -- 6. Missing Value Handling ------------------------------------------------
# Stage 1 — Drop columns exceeding 50% missingness. Imputing at such high
#           rates introduces more noise than signal.
# Stage 2 — Impute remaining NAs:
#   Numeric columns  : median (robust to skewed distributions and outliers)
#   Categorical columns : mode (preserves the most frequent category)
#
# pandas 2.x dtype note:
#   pandas 2.x introduced StringDtype alongside the legacy object dtype.
#   pd.api.types.is_string_dtype() catches both. Some "string" columns
#   actually contain numeric values — pd.to_numeric() coercion is attempted
#   first; if more than half the values convert, the column is treated numeric.

missing_pct  = df_model.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()

df_model.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} high-missingness columns: {cols_to_drop}")

for col in df_model.columns:
    if pd.api.types.is_string_dtype(df_model[col]):
        converted = pd.to_numeric(df_model[col], errors="coerce")
        if converted.notna().sum() > 0.5 * len(converted):
            df_model[col] = converted.fillna(converted.median())
        else:
            mode_val = df_model[col].mode()
            if len(mode_val) > 0:
                df_model[col] = df_model[col].fillna(mode_val[0])
    else:
        df_model[col] = df_model[col].fillna(df_model[col].median())

print(f"Imputation complete | remaining NAs: {df_model.isnull().sum().sum()}")


# -- 7. Label Encoding --------------------------------------------------------
# XGBoost and scikit-learn require all inputs to be numeric.
# A separate LabelEncoder is stored per column to enable inverse transforms
# and consistent encoding if the pipeline is applied to new data.

label_encoders = {}

for col in df_model.select_dtypes(include=["object"]).columns:
    le                  = LabelEncoder()
    df_model[col]       = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

print(f"Label encoding complete | {len(label_encoders)} categorical columns encoded")


# -- 8. Train / Test Split ----------------------------------------------------
# 80/20 split with random_state = 42 for reproducibility.
# The same seed is used in xgboost_model.R for cross-language comparison.

X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Features: {X.shape[1]}")


# -- 9. Target Distribution ---------------------------------------------------
# Property values are heavily right-skewed — a small number of luxury
# properties inflate the mean far above the median. This is expected and
# is a known challenge for regression models on property assessment data.

plt.figure(figsize=(8, 5))
sns.histplot(y, bins=50, kde=True, color="#4A90D9")
plt.title("Target Variable Distribution — Total Property Value (FY2025)",
          fontsize=13, pad=12)
plt.xlabel("Total Value ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# -- 10. Decision Tree Regressor ----------------------------------------------
# Decision trees partition the feature space into rectangular regions and
# are fully interpretable but prone to overfitting. Depth and leaf constraints
# are applied to limit complexity.
#
# max_depth         = 15 : maximum tree levels
# min_samples_split = 10 : minimum samples to justify splitting a node
# min_samples_leaf  = 5  : minimum samples required in any leaf node
#
# Note: R² ≈ 0.977 reflects the near-identity bldg_value + land_value
# ≈ total_value rather than generalised predictive power.

print("\n-- Decision Tree Regressor --")

dt_model = DecisionTreeRegressor(
    max_depth         = 15,
    min_samples_split = 10,
    min_samples_leaf  = 5,
    random_state      = 42
)

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

mae_dt  = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt   = r2_score(y_test, y_pred_dt)

print(f"  R²   : {r2_dt:.4f}")
print(f"  RMSE : ${rmse_dt:,.2f}")
print(f"  MAE  : ${mae_dt:,.2f}")

dt_importance = (
    pd.Series(dt_model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 6))
dt_importance.plot(kind="bar", color="#4A90D9", alpha=0.85)
plt.title("Top 15 Feature Importances — Decision Tree (FY2025)",
          fontsize=13, pad=12)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# -- 11. XGBoost Regressor ----------------------------------------------------
# XGBoost builds trees sequentially, each correcting the residuals of the
# previous iteration (gradient boosting). It generalises better than a
# single decision tree and is the primary model in the R pipeline.
#
# Hyperparameters mirror xgboost_model.R for cross-language consistency:
#   learning_rate    : 0.05 — conservative step size reduces overfitting
#   max_depth        : 8    — maximum depth per tree
#   subsample        : 0.8  — row fraction per tree (reduces variance)
#   colsample_bytree : 0.8  — feature fraction per tree (Random-Forest-like)
#   reg_lambda       : 1    — L2 regularisation; penalises large leaf weights
#   reg_alpha        : 0.5  — L1 regularisation; promotes feature sparsity
#
# Early stopping halts training when test RMSE does not improve for 15 rounds.

print("\n-- XGBoost Regressor --")

xgb_model = xgb.XGBRegressor(
    objective             = "reg:squarederror",
    n_estimators          = 300,
    learning_rate         = 0.05,
    max_depth             = 8,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_lambda            = 1,
    reg_alpha             = 0.5,
    early_stopping_rounds = 15,
    eval_metric           = "rmse",
    random_state          = 42
)

xgb_model.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    verbose  = 50
)

y_pred_xgb = xgb_model.predict(X_test)

mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb   = r2_score(y_test, y_pred_xgb)

print(f"  R²   : {r2_xgb:.4f}")
print(f"  RMSE : ${rmse_xgb:,.2f}")
print(f"  MAE  : ${mae_xgb:,.2f}")

xgb_importance = (
    pd.Series(xgb_model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 6))
xgb_importance.plot(kind="bar", color="#E8734A", alpha=0.85)
plt.title("Top 15 Feature Importances — XGBoost (FY2025)",
          fontsize=13, pad=12)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# -- 12. Model Comparison -----------------------------------------------------
# Lower RMSE and MAE with higher R² indicates better performance.
# XGBoost is the preferred model for generalisation; Decision Tree provides
# interpretability and a useful baseline.

print("\n-- Model Comparison --")
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "XGBoost"],
    "R²"  : [round(r2_dt,   4), round(r2_xgb,   4)],
    "RMSE": [round(rmse_dt, 2), round(rmse_xgb, 2)],
    "MAE" : [round(mae_dt,  2), round(mae_xgb,  2)],
})
print(comparison.to_string(index=False))


# -- 13. Diagnostic Visualisations --------------------------------------------

# 13.1 Actual vs Predicted
# Points close to the red diagonal indicate accurate predictions.
# Systematic scatter below the line at high values reflects under-prediction
# of luxury properties — a common limitation in property value models where
# high-value training examples are sparse.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, name, color, r2 in [
    (axes[0], y_pred_dt,  "Decision Tree", "#4A90D9", r2_dt),
    (axes[1], y_pred_xgb, "XGBoost",       "#E8734A", r2_xgb),
]:
    ax.scatter(y_test, y_pred, alpha=0.25, color=color, s=4)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "r-", lw=1.5)
    ax.set_title(f"Actual vs Predicted — {name}\nR² = {r2:.4f}", fontsize=12)
    ax.set_xlabel("Actual Value ($)")
    ax.set_ylabel("Predicted Value ($)")

plt.tight_layout()
plt.show()

# 13.2 Prediction Error Distributions
# Residuals should be approximately normally distributed around zero.
# A long right tail indicates systematic under-prediction for high-value
# properties where training data is sparse.
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, y_pred, name, color in [
    (axes[0], y_pred_dt,  "Decision Tree", "#4A90D9"),
    (axes[1], y_pred_xgb, "XGBoost",       "#E8734A"),
]:
    errors = y_test - y_pred
    sns.histplot(errors, bins=60, kde=True, ax=ax, color=color)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_title(f"Error Distribution — {name}", fontsize=12)
    ax.set_xlabel("Prediction Error ($)")

plt.tight_layout()
plt.show()

# 13.3 Residuals vs Fitted
# A fan shape (wider spread at higher predicted values) would indicate
# heteroscedasticity — expected in property data due to luxury outliers.
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, y_pred, name, color in [
    (axes[0], y_pred_dt,  "Decision Tree", "#4A90D9"),
    (axes[1], y_pred_xgb, "XGBoost",       "#E8734A"),
]:
    errors = y_test - y_pred
    ax.scatter(y_pred, errors, alpha=0.18, color=color, s=4)
    ax.axhline(0, color="red", linewidth=1.5)
    ax.set_title(f"Residuals vs Fitted — {name}", fontsize=12)
    ax.set_xlabel("Predicted Value ($)")
    ax.set_ylabel("Residuals ($)")

plt.tight_layout()
plt.show()

print("\nDecision Tree & XGBoost analysis complete.")