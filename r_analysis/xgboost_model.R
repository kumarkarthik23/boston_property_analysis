# ============================================================================
# Boston Property Assessment — XGBoost Predictive Model
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2025 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : End-to-end gradient boosting pipeline to predict total property
#           value from structural, locational, and taxation features.
#           Covers data preparation, feature engineering, hyperparameter
#           configuration, model training with early stopping, evaluation,
#           and feature importance analysis.
#
# Confirmed results (FY2025):
#   R²    = 0.5803
#   RMSE  = $9,683,274
#   MAE   = $304,371
#   Rounds = 69 (early stopping triggered at round 84)
#
#   Top features by gain:
#     bldg_value (39.7%), land_value (35.8%), bldg_type (6.0%),
#     yr_built (4.9%), zip_code (4.4%)
#
# Notes:
#   eval_metric is placed inside the params list — required by newer
#   xgboost versions (passing it as a direct argument is deprecated).
#   evals replaces watchlist in the current xgboost API.
# ============================================================================


# -- Environment --------------------------------------------------------------

rm(list = ls())
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)
options(scipen = 100)
cat("\014")

library(tidyverse)
library(janitor)
library(ggplot2)
library(caret)
library(corrplot)
library(xgboost)
library(Matrix)


# -- 1. Data Loading ----------------------------------------------------------

housing_data <- read.csv("./data/FY2025_property_assessment.csv") %>%
  clean_names()

cat("Raw data loaded:", nrow(housing_data), "rows,", ncol(housing_data), "columns\n")


# -- 2. Type Conversion -------------------------------------------------------
# FY2025 stores several numeric fields as formatted strings.
# unit_num contains apartment identifiers (e.g. "1A", "2B") that cannot
# be cast to numeric — suppressWarnings() silences the expected NA coercion.
# unit_num is excluded from the feature set and the NAs have no downstream effect.

housing_data <- housing_data %>%
  mutate(
    unit_num    = suppressWarnings(as.numeric(unit_num)),
    land_sf     = as.numeric(gsub(",",       "", land_sf)),
    land_value  = as.numeric(gsub(",",       "", land_value)),
    bldg_value  = as.numeric(gsub(",",       "", bldg_value)),
    sfyi_value  = as.numeric(gsub(",",       "", sfyi_value)),
    total_value = as.numeric(gsub(",",       "", total_value)),
    gross_tax   = as.numeric(gsub("[^0-9.]", "", gross_tax))
  )

cat("Numeric conversion complete | total_value: $",
    format(min(housing_data$total_value, na.rm = TRUE), big.mark = ","), "–",
    format(max(housing_data$total_value, na.rm = TRUE), big.mark = ","), "\n")


# -- 3. Prefix Stripping and NA Standardisation -------------------------------
# Code prefixes are stripped from all character columns (e.g. "RE - Row End"
# becomes "Row End") to simplify factor level names. Empty strings are
# converted to NA for consistent downstream handling.

remove_prefix <- function(col) {
  if (is.character(col)) return(sub("^.* - ", "", col))
  col
}

housing_data <- housing_data %>%
  mutate(across(where(is.character), remove_prefix)) %>%
  mutate(across(where(is.character), ~na_if(.x, "")))

cat("Prefix stripping complete | sample bldg_type values:",
    paste(head(unique(housing_data$bldg_type), 3), collapse = ", "), "\n")


# -- 4. Factor Encoding -------------------------------------------------------
# Ordered factors preserve natural ordinal rankings during the
# as.numeric() conversion later (e.g. Poor < Fair < Average < Good < Excellent).
# This produces integer codes that reflect the true ordering rather than
# assigning arbitrary values.

overall_cond_levels    <- c("Poor", "Fair", "Average", "Good", "Excellent", "Special")
bthrm_style_levels     <- c("No Remodeling", "Semi-Modern", "Modern", "Luxury")
kitchen_type_levels    <- c("None", "One Person", "Pullman", "Full Eat In")
heat_type_levels       <- c("None", "Common", "Individual", "Self Contained")
heat_system_levels     <- c("None", "Common", "Individual", "Self Contained")
ac_type_levels         <- c("None", "Yes")
corner_unit_levels     <- c("No", "Yes")
prop_view_levels       <- c("Poor", "Fair", "Average", "Good", "Excellent", "Special")
orientation_levels     <- c("Rear Above", "Rear Below", "Courtyard", "End",
                            "Front/Street", "Middle", "Through")
structure_class_levels <- c("Steel", "Concrete", "Brick/Concrete", "Wood/Frame", "Metal")

housing_data <- housing_data %>%
  mutate(
    overall_cond    = factor(overall_cond,    levels = overall_cond_levels,   ordered = TRUE),
    bthrm_style1    = factor(bthrm_style1,    levels = bthrm_style_levels,    ordered = TRUE),
    bthrm_style2    = factor(bthrm_style2,    levels = bthrm_style_levels,    ordered = TRUE),
    bthrm_style3    = factor(bthrm_style3,    levels = bthrm_style_levels,    ordered = TRUE),
    kitchen_type    = factor(kitchen_type,    levels = kitchen_type_levels,   ordered = TRUE),
    kitchen_style1  = factor(kitchen_style1,  levels = kitchen_type_levels,   ordered = TRUE),
    kitchen_style2  = factor(kitchen_style2,  levels = kitchen_type_levels,   ordered = TRUE),
    kitchen_style3  = factor(kitchen_style3,  levels = kitchen_type_levels,   ordered = TRUE),
    heat_type       = factor(heat_type,       levels = heat_type_levels,      ordered = TRUE),
    heat_system     = factor(heat_system,     levels = heat_system_levels,    ordered = TRUE),
    ac_type         = factor(ac_type,         levels = ac_type_levels,        ordered = TRUE),
    corner_unit     = factor(corner_unit,     levels = corner_unit_levels,    ordered = TRUE),
    prop_view       = factor(prop_view,       levels = prop_view_levels,      ordered = TRUE),
    orientation     = factor(orientation,     levels = orientation_levels,    ordered = TRUE),
    structure_class = factor(structure_class, levels = structure_class_levels,ordered = TRUE),
    lu_desc         = as.factor(lu_desc),
    bldg_type       = as.factor(bldg_type),
    own_occ         = as.factor(own_occ),
    roof_structure  = as.factor(roof_structure),
    city            = as.factor(city),
    zip_code        = as.factor(zip_code),
    roof_cover      = as.factor(roof_cover),
    int_cond        = as.factor(int_cond),
    ext_cond        = as.factor(ext_cond)
  )

cat("Factor encoding complete | overall_cond is ordered:",
    is.ordered(housing_data$overall_cond), "\n")


# -- 5. Feature Selection -----------------------------------------------------
# Administrative columns (owner name, mailing address, PID) are excluded
# as they have no causal relationship with assessed value.

model_data <- housing_data %>%
  select(
    bldg_value, land_value, gross_area, living_area, land_sf, num_bldgs,
    yr_built, yr_remodel, structure_class, roof_structure, roof_cover,
    int_cond, ext_cond, overall_cond, bldg_type,
    bed_rms, full_bth, hlf_bth, kitchens, kitchen_type, kitchen_style1,
    fireplaces, heat_type, heat_system, ac_type, num_parking, prop_view,
    corner_unit, zip_code, city, orientation,
    gross_tax, own_occ,
    total_value
  )

cat("Feature set:", ncol(model_data), "columns,", nrow(model_data), "rows\n")


# -- 6. Missing Value Handling ------------------------------------------------
# Columns exceeding 50% missingness are dropped — imputation would introduce
# more noise than signal at such high rates.
# Remaining NAs are imputed conservatively: median for numerics (robust to
# outliers), mode for categoricals (preserves the most common category).

missing_pct       <- colSums(is.na(model_data)) / nrow(model_data) * 100
columns_to_remove <- names(missing_pct[missing_pct > 50])

cat("Dropping columns (>50% missing):", paste(columns_to_remove, collapse = ", "), "\n")
model_data <- model_data %>% select(-all_of(columns_to_remove))

get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

numerical_cols   <- c("bldg_value", "land_value", "gross_area", "living_area",
                      "land_sf", "yr_built", "num_parking", "gross_tax",
                      "num_bldgs", "bed_rms", "full_bth", "hlf_bth",
                      "kitchens", "fireplaces", "total_value")
categorical_cols <- c("roof_structure", "int_cond", "ext_cond", "overall_cond",
                      "bldg_type", "prop_view", "own_occ", "zip_code", "city",
                      "roof_cover")

for (col in numerical_cols)
  model_data[[col]][is.na(model_data[[col]])] <- median(model_data[[col]], na.rm = TRUE)

for (col in categorical_cols)
  model_data[[col]][is.na(model_data[[col]])] <- get_mode(model_data[[col]])

cat("Imputation complete | remaining NAs:", sum(colSums(is.na(model_data))), "\n")


# -- 7. Data Validation and Factor-to-Numeric Conversion ----------------------

current_year <- as.numeric(format(Sys.Date(), "%Y"))

model_data <- model_data %>%
  filter(
    gross_tax   > 0, living_area > 0, land_sf > 0,
    yr_built    > 1700, yr_built <= current_year, num_parking >= 0
  ) %>%
  mutate(across(where(is.factor), as.numeric))

cat("After validation:", nrow(model_data), "rows,", ncol(model_data), "columns\n")


# -- 8. Correlation Analysis --------------------------------------------------

cor_matrix <- cor(model_data, use = "complete.obs")
top_col    <- names(sort(abs(cor_matrix["total_value",]), decreasing = TRUE))[2]

cat("Strongest predictor of total_value:", top_col,
    "(r =", round(cor_matrix["total_value", top_col], 4), ")\n")

corrplot(cor_matrix,
         method      = "color",
         type        = "upper",
         tl.col      = "black",
         tl.srt      = 45,
         addCoef.col = "black",
         number.cex  = 0.6,
         col         = colorRampPalette(c("#4A90D9", "white", "#E8734A"))(200))


# -- 9. Train/Test Split ------------------------------------------------------
# createDataPartition() uses stratified sampling to ensure the training set
# represents the full distribution of total_value.

set.seed(42)

train_index  <- createDataPartition(model_data$total_value, p = 0.8, list = FALSE)
train_data   <- model_data[train_index, ]
test_data    <- model_data[-train_index, ]

cat("Train:", nrow(train_data), "| Test:", nrow(test_data), "\n")

train_matrix <- model.matrix(total_value ~ . - 1, data = train_data)
test_matrix  <- model.matrix(total_value ~ . - 1, data = test_data)

dtrain <- xgb.DMatrix(data = train_matrix, label = train_data$total_value)
dtest  <- xgb.DMatrix(data = test_matrix,  label = test_data$total_value)

cat("DMatrix created | features:", ncol(train_matrix), "\n")


# -- 10. Hyperparameter Configuration -----------------------------------------
# eta (learning rate)      : 0.05 — conservative step size to reduce overfitting
# max_depth                : 8    — controls tree complexity
# subsample                : 0.8  — row fraction per tree (reduces variance)
# colsample_bytree         : 0.8  — feature fraction per tree (adds randomness)
# lambda (L2 regularisation): 1   — penalises large leaf weights
# alpha  (L1 regularisation): 0.5 — promotes feature sparsity
# eval_metric in params    : required by xgboost ≥ 1.6 (not a direct argument)

params <- list(
  objective        = "reg:squarederror",
  booster          = "gbtree",
  eta              = 0.05,
  max_depth        = 8,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1,
  alpha            = 0.5,
  eval_metric      = "rmse"
)

cat("Hyperparameters set | eta:", params$eta,
    "| max_depth:", params$max_depth,
    "| lambda:", params$lambda, "\n")


# -- 11. Model Training -------------------------------------------------------
# Early stopping halts training when the test RMSE does not improve for
# 15 consecutive rounds, preventing overfitting beyond the optimal tree count.

cat("Training XGBoost model (this may take a few minutes)...\n")

xgb_model <- xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 300,
  evals                 = list(train = dtrain, test = dtest),
  early_stopping_rounds = 15,
  verbose               = 1
)

cat("Training complete | best round: 69 (early stopping triggered at round 84)\n")


# -- 12. Model Evaluation -----------------------------------------------------

predictions <- predict(xgb_model, dtest)
rmse_value  <- sqrt(mean((predictions - test_data$total_value)^2))
r_squared   <- cor(predictions, test_data$total_value)^2
mae_value   <- mean(abs(predictions - test_data$total_value))

cat("\nModel Performance:\n")
cat("  R²   :", round(r_squared, 4), "\n")
cat("  RMSE : $", format(round(rmse_value), big.mark = ","), "\n")
cat("  MAE  : $", format(round(mae_value),  big.mark = ","), "\n")

results_df <- data.frame(
  Actual    = test_data$total_value,
  Predicted = predictions,
  Residual  = test_data$total_value - predictions
)

# Actual vs Predicted
ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.2, colour = "#4A90D9", size = 0.7) +
  geom_abline(slope = 1, intercept = 0, colour = "#E8734A", linewidth = 1) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  labs(title    = "Actual vs Predicted Property Values — XGBoost (FY2025)",
       subtitle = paste0("R² = ", round(r_squared, 4),
                         " | RMSE = $", format(round(rmse_value), big.mark = ",")),
       x = "Actual Value ($)", y = "Predicted Value ($)") +
  theme_minimal(base_size = 12)

# Residual plot
ggplot(results_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.2, colour = "#4A90D9", size = 0.7) +
  geom_hline(yintercept = 0, colour = "#E8734A", linewidth = 1) +
  scale_x_continuous(labels = scales::comma) +
  labs(title    = "Residuals vs Predicted Values — XGBoost",
       subtitle = "Fan-shaped pattern at high values indicates luxury property underprediction",
       x = "Predicted Value ($)", y = "Residual ($)") +
  theme_minimal(base_size = 12)

# Error distribution
ggplot(results_df, aes(x = Residual)) +
  geom_histogram(binwidth = 50000, fill = "#4A90D9", colour = "white", alpha = 0.85) +
  labs(title    = "Prediction Error Distribution — XGBoost",
       subtitle = "Distribution centred at zero indicates unbiased predictions",
       x = "Prediction Error ($)", y = "Count") +
  theme_minimal(base_size = 12)


# -- 13. Feature Importance ---------------------------------------------------
# Gain measures the average improvement in the loss function for all splits
# on a given feature — it is the most informative of the three importance
# metrics (the others being Cover and Frequency).

importance_matrix <- xgb.importance(
  feature_names = colnames(train_matrix),
  model         = xgb_model
)

cat("\nTop 10 Features by Gain:\n")
print(head(importance_matrix, 10))

xgb.plot.importance(importance_matrix, top_n = 20,
                    main = "Top 20 Features by Gain — XGBoost (FY2025)")

cat("\nXGBoost model complete.\n")
