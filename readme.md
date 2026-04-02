# Greater Boston Property Valuation Analysis

An end-to-end data analytics and machine learning project analysing **183,445 property assessment records** across Boston's fiscal years 2023 and 2025. The project covers exploratory data analysis, year-over-year comparative analysis, statistical hypothesis testing, regression modelling, gradient boosting, and large-scale distributed processing — implemented across R and Python.

An interactive Tableau dashboard accompanies this project:
**[Greater Boston Property Analysis — Tableau Public](https://public.tableau.com/app/profile/kumarkarthik23/viz/GreaterBostonPropertyAnalysis/GreaterBostonPropertyAnalysis)**

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Model Results](#model-results)
- [Data Sources](#data-sources)

---

## Overview

Boston property values and tax assessments changed significantly between FY2023 and FY2025. This project quantifies those changes, identifies which property features drive value, and builds predictive models to estimate assessed value from structural and locational features.

**Research questions addressed:**
1. How did property values change between FY2023 and FY2025?
2. Did the gross tax burden increase relative to assessed value?
3. Do bedroom and bathroom counts command a measurable value premium?
4. Has the distribution of building types shifted?
5. Which structural features are the strongest predictors of assessed value?

---

## Key Findings

### Year-over-Year Changes (FY2023 → FY2025)

| Metric | FY2023 | FY2025 | Change |
|--------|--------|--------|--------|
| Total records | 178,598 | 183,445 | +2.7% |
| Median total value | $574,500 | $661,000 | +15.1% |
| Median gross tax | $6,162 | $8,157 | +32.4% |
| Top ZIP avg value | $6.14M (02210) | $9.73M (02199) | +58.5% |
| Tax-to-value ratio | 1.088% | 1.158% | +6.4pp |

### Matched Property Analysis (63,667 properties in both years)

| Metric | Change |
|--------|--------|
| Median property value | +17.34% |
| Median gross tax | +24.88% |
| Median land value | +18.63% |
| Median building value | +15.57% |

> Taxes rose **44% faster** than property values on matched properties — the central finding supporting a taxation fairness concern.

### Hypothesis Testing Results (α = 0.05)

| Hypothesis | Result |
|-----------|--------|
| H1: Property values changed 2023–2025 | ✅ Reject H0 (p = 0.000003) |
| H2: Gross tax increased 2023–2025 | ❌ Fail to Reject (composition effect — see H5) |
| H3: 3+ bedrooms command higher values | ✅ Reject H0 (p ≈ 0) |
| H4: Owner-occupied differ in value | ✅ Reject H0 (p ≈ 0) |
| H5: Tax-to-value ratio increased | ✅ Reject H0 (p ≈ 0) |
| ANOVA: Building type affects value | ✅ Reject H0 (F = 510.99, 31 types) |

> H2 fails due to portfolio composition — FY2023 contains proportionally more commercial properties inflating the mean. H5 (tax-to-value ratio) is the more reliable measure and confirms the tax burden increased significantly.

### Regression Results

| Model | Adj R² |
|-------|--------|
| SLR — Land SF (FY2023) | 0.032 |
| SLR — Land SF (FY2025) | 0.100 |
| SLR — Living Area (FY2023) | 0.504 |
| SLR — Living Area (FY2025) | 0.341 |
| MLR — 8 features (FY2023) | 0.565 |
| MLR — 8 features (FY2025) | 0.536 |
| MLR — with building type dummies (FY2025) | **0.656** |

Key coefficient shifts FY2023 → FY2025:
- `log_living_area`: +0.148 — interior space commands a larger premium in 2025
- `age_of_property`: −0.001 — older properties lost their relative premium
- `log_land_sf`: −0.138 — land became a negative predictor in Boston's dense urban market

### PySpark Analysis (FY2025)

| Metric | Value |
|--------|-------|
| Highest avg value city | Boston ($3.25M, n = 48,036) |
| Highest tax burden city | Roxbury (1.307% avg tax-to-value ratio) |
| Mean price per sq ft | $586.37 (stddev $375.49) |
| Unique property types | 194 |
| Duplicate rows | 0 |

---

## Project Structure

```
boston_property_analysis/
├── data/
│   ├── FY2023_property_assessment.csv
│   ├── FY2025_property_assessment.csv
│   └── data_sources.md
├── r_analysis/
│   ├── eda_2023.R                    # Exploratory analysis — FY2023
│   ├── eda_2025.R                    # Exploratory analysis — FY2025
│   ├── comparative_analysis.R        # YoY comparison across 5 research questions
│   ├── hypothesis_tests.R            # Formal hypothesis testing (H1–H5 + ANOVA)
│   ├── regression_analysis.R         # SLR, MLR, and building type dummies
│   └── xgboost_model.R               # Gradient boosting model
├── python_analysis/
│   ├── decision_tree_xgboost_model.py  # Decision Tree and XGBoost pipeline
│   └── pyspark_analysis.py             # Distributed analysis with Apache Spark
└── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Languages | R, Python |
| Data Wrangling | tidyverse, janitor (R) · pandas, numpy (Python) |
| Visualisation | ggplot2, corrplot (R) · matplotlib, seaborn (Python) · Tableau |
| Machine Learning | xgboost, caret (R) · scikit-learn, xgboost (Python) |
| Distributed Processing | Apache Spark 4.1.1 via PySpark |
| Environment | RStudio · VS Code · macOS (Apple Silicon) |

---

## How to Run

### Prerequisites

**R packages:**
```r
install.packages(c("tidyverse", "janitor", "ggplot2", "scales",
                   "corrplot", "caret", "xgboost", "Matrix"))
```

**Python packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost pyspark
```

**PySpark additional requirements:**
- Java 17 or higher (`brew install openjdk@17` on Mac)
- OpenMP for XGBoost on Mac (`brew install libomp`)

---

### Running R Scripts

All R scripts read data from `../data/` relative to the `r_analysis/` folder. Before running any script, set the working directory to the script's location:

**RStudio → Session → Set Working Directory → To Source File Location**

Run scripts in the following order:

```
eda_2023.R
eda_2025.R
comparative_analysis.R
hypothesis_tests.R
regression_analysis.R
xgboost_model.R
```

---

### Running Python Scripts

Open a terminal in the project root and navigate to `python_analysis/`:

```bash
cd python_analysis
python3 decision_tree_xgboost_model.py
python3 pyspark_analysis.py
```

---

## Model Results

### R XGBoost (xgboost_model.R)

| Metric | Value |
|--------|-------|
| R² | 0.5803 |
| RMSE | $9,683,274 |
| MAE | $304,371 |
| Best round | 69 (early stopping triggered at round 84) |

Top features by gain: `bldg_value` (39.7%), `land_value` (35.8%), `bldg_type` (6.0%), `yr_built` (4.9%), `zip_code` (4.4%)

---

### Python Models (decision_tree_xgboost_model.py)

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| Decision Tree | 0.9777 | $1,876,075 | $67,641 |
| XGBoost | 0.7489 | $6,290,224 | $299,554 |

> The Decision Tree's high R² reflects the near-identity `bldg_value + land_value ≈ total_value` in the assessment data rather than generalised predictive power. The XGBoost MAE of $299,554 closely aligns with the R pipeline's $304,371, confirming cross-language consistency.

**Cross-language comparison (Python vs R XGBoost):**

| | Python | R |
|-|--------|---|
| R² | 0.7489 | 0.5803 |
| RMSE | $6.29M | $9.68M |
| MAE | $299,554 | $304,371 |

Python performs better on RMSE because it trains on more records (no IQR outlier filtering). The near-identical MAE values confirm both pipelines are consistent at the typical-property level.

---

## Data Sources

| Dataset | Source | Records |
|---------|--------|---------|
| FY2023 Property Assessment | [Analyze Boston](https://data.boston.gov/dataset/property-assessment) | 178,598 |
| FY2025 Property Assessment | [Analyze Boston](https://data.boston.gov/dataset/property-assessment) | 183,445 |

Data is publicly available from the City of Boston's open data portal and refreshed annually.

---

*Author: Kumar Karthik Ankasandra Naveen · MPS Analytics, Northeastern University*