# ============================================================================
# Boston Property Assessment — Regression Analysis
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2023 and FY2025 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : Simple and multiple linear regression models on both assessment
#           years, with a year-over-year coefficient comparison to identify
#           how the marginal value of each property feature shifted.
#
# Models:
#   1. Simple regression: Log(Value) ~ Log(Land SF)       — FY2023 & FY2025
#   2. Simple regression: Log(Value) ~ Log(Living Area)   — FY2023 & FY2025
#   3. Multiple regression: 8 structural features         — FY2025
#   4. Multiple regression: 8 structural features         — FY2023
#   5. Multiple regression + building type dummies        — FY2025
#      Reference level: Colonial (most common type after prefix stripping)
#   6. Coefficient comparison: FY2023 vs FY2025
#   7. Residual diagnostics: FY2025 multiple regression
#
# Confirmed results:
#   SLR Land SF       : FY2023 R² = 0.032 | FY2025 R² = 0.100
#   SLR Living Area   : FY2023 R² = 0.504 | FY2025 R² = 0.341
#   MLR 8 features    : FY2023 Adj R² = 0.565 | FY2025 Adj R² = 0.536
#   MLR + type dummies: FY2025 Adj R² = 0.656 (+12pp over base MLR)
# ============================================================================


# -- Environment --------------------------------------------------------------

rm(list = ls())
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)
options(scipen = 100)
cat("\014")

library(tidyverse)
library(janitor)
library(ggplot2)
library(scales)


# -- 1. Shared Data Loading Function ------------------------------------------
# Log transformations are applied during loading so they are available
# throughout all model sections. Coefficients in log–log models are
# interpreted as elasticities: a 1% change in X is associated with a
# β% change in Y, holding all other variables constant.
#
# Type alignment (across(everything(), as.character)):
#   FY2023 stores st_num as character; FY2025 stores it as integer.
#   Flattening all columns to character before numeric re-conversion
#   ensures bind_rows() succeeds without type conflict errors.
#
# Targeted NA filter:
#   na.omit() on the full dataframe removes all rows because sparse
#   columns (kitchen_style, heat_system, orientation) are NA for nearly
#   every record. We filter only on columns the regression uses.

load_and_clean <- function(filepath, year) {
  read.csv(filepath) %>%
    clean_names() %>%
    mutate(across(everything(), as.character)) %>%
    mutate(
      land_sf     = as.numeric(gsub(",",       "", land_sf)),
      land_value  = as.numeric(gsub(",",       "", land_value)),
      bldg_value  = as.numeric(gsub(",",       "", bldg_value)),
      total_value = as.numeric(gsub(",",       "", total_value)),
      gross_tax   = as.numeric(gsub("[^0-9.]", "", gross_tax)),
      living_area = as.numeric(gsub(",",       "", living_area)),
      gross_area  = as.numeric(gsub(",",       "", gross_area)),
      yr_built    = as.numeric(yr_built),
      yr_remodel  = as.numeric(yr_remodel),
      bed_rms     = as.numeric(bed_rms),
      full_bth    = as.numeric(full_bth),
      hlf_bth     = as.numeric(hlf_bth),
      num_parking = as.numeric(num_parking)
    ) %>%
    filter(
      total_value > 0, gross_tax   > 0,
      living_area > 0, land_sf     > 0,
      !is.na(yr_built), yr_built > 1700,
      yr_built <= as.numeric(format(Sys.Date(), "%Y")),
      bed_rms  >= 0, bed_rms  <= 10,
      full_bth >= 0, full_bth <= 8
    ) %>%
    mutate(
      fiscal_year        = year,
      log_total_value    = log(total_value),
      log_land_sf        = log(land_sf),
      log_living_area    = log(living_area),
      age_of_property    = year - yr_built,
      price_per_sqft     = total_value / living_area,
      tax_to_value_ratio = gross_tax / total_value * 100,
      bldg_type_clean    = sub("^.* - ", "", bldg_type)
    ) %>%
    filter(
      !is.na(log_total_value), !is.na(log_land_sf), !is.na(log_living_area),
      !is.na(bed_rms), !is.na(full_bth), !is.na(hlf_bth),
      !is.na(num_parking), !is.na(age_of_property), !is.na(bldg_type_clean)
    )
}

df2023 <- load_and_clean("./data/FY2023_property_assessment.csv", 2023)
df2025 <- load_and_clean("./data/FY2025_property_assessment.csv", 2025)

cat("FY2023:", nrow(df2023), "records | FY2025:", nrow(df2025), "records\n")

combined <- bind_rows(df2023, df2025) %>%
  mutate(fiscal_year = as.factor(fiscal_year))

cat("Combined dataset:", nrow(combined), "records\n")

# Auto-detect the fireplaces column name — FY2025 uses "fireplaces",
# FY2023 uses "fire_place" (the column was renamed between assessments)
fp_col_2023 <- if ("fireplaces" %in% names(df2023)) "fireplaces" else "fire_place"
fp_col_2025 <- if ("fireplaces" %in% names(df2025)) "fireplaces" else "fire_place"
cat("Fireplaces column | FY2023:", fp_col_2023, "| FY2025:", fp_col_2025, "\n")


# -- 2. Simple Regression: Value ~ Land SF ------------------------------------
# The slope coefficient is the land-size elasticity of value: a 1% increase
# in land area is associated with approximately β% increase in assessed value.
# The higher R² and steeper slope in FY2025 confirms that land commanded a
# larger premium in 2025 than in 2023.

cat("\n── Simple Regression: Log(Value) ~ Log(Land SF) ──\n")

slr_land_2023 <- lm(log_total_value ~ log_land_sf, data = df2023)
slr_land_2025 <- lm(log_total_value ~ log_land_sf, data = df2025)

cat("FY2023 | R² =", round(summary(slr_land_2023)$r.squared, 4),
    "| slope =", round(coef(slr_land_2023)["log_land_sf"], 4), "\n")
cat("FY2025 | R² =", round(summary(slr_land_2025)$r.squared, 4),
    "| slope =", round(coef(slr_land_2025)["log_land_sf"], 4), "\n")

cat("\nFY2023 model:\n"); print(summary(slr_land_2023))
cat("\nFY2025 model:\n"); print(summary(slr_land_2025))

ggplot(combined, aes(x = log_land_sf, y = log_total_value, colour = fiscal_year)) +
  geom_point(alpha = 0.12, size = 0.6) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  scale_colour_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  labs(title    = "Log Property Value vs Log Land Size: FY2023 vs FY2025",
       subtitle = "Slope = land-size elasticity of assessed value",
       x = "Log(Land Area, sq ft)", y = "Log(Total Value)", colour = "Year") +
  theme_minimal(base_size = 12)


# -- 3. Simple Regression: Value ~ Living Area --------------------------------
# Living area explains substantially more variance than land size (R² ≈ 0.50
# vs 0.03 in FY2023), reflecting the importance of interior space in
# dense urban markets. The weaker R² in FY2025 (0.34) is attributable to
# the more diverse property mix in the larger FY2025 dataset.

cat("\n── Simple Regression: Log(Value) ~ Log(Living Area) ──\n")

slr_living_2023 <- lm(log_total_value ~ log_living_area, data = df2023)
slr_living_2025 <- lm(log_total_value ~ log_living_area, data = df2025)

cat("FY2023 | R² =", round(summary(slr_living_2023)$r.squared, 4),
    "| slope =", round(coef(slr_living_2023)["log_living_area"], 4), "\n")
cat("FY2025 | R² =", round(summary(slr_living_2025)$r.squared, 4),
    "| slope =", round(coef(slr_living_2025)["log_living_area"], 4), "\n")

cat("\nFY2023 model:\n"); print(summary(slr_living_2023))
cat("\nFY2025 model:\n"); print(summary(slr_living_2025))

ggplot(combined, aes(x = log_living_area, y = log_total_value, colour = fiscal_year)) +
  geom_point(alpha = 0.12, size = 0.6) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  scale_colour_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  labs(title    = "Log Property Value vs Log Living Area: FY2023 vs FY2025",
       subtitle = "Slope = living-area elasticity of assessed value",
       x = "Log(Living Area, sq ft)", y = "Log(Total Value)", colour = "Year") +
  theme_minimal(base_size = 12)


# -- 4. Multiple Regression — FY2025 ------------------------------------------
# Controlling for all eight features simultaneously isolates the independent
# contribution of each variable after accounting for the others.
# Adj R² ≈ 0.536 — a substantial improvement over both simple regressions.

cat("\n── Multiple Regression (8 features) — FY2025 ──\n")

mlr_2025 <- lm(
  as.formula(paste("log_total_value ~ log_land_sf + log_living_area + bed_rms +",
                   "full_bth + hlf_bth +", fp_col_2025, "+ num_parking + age_of_property")),
  data = df2025
)

cat("Adj R² =", round(summary(mlr_2025)$adj.r.squared, 4), "\n")
cat("Significant predictors (p < 0.05):",
    sum(summary(mlr_2025)$coefficients[-1, 4] < 0.05), "of",
    nrow(summary(mlr_2025)$coefficients) - 1, "\n")

print(summary(mlr_2025))

coef_df <- as.data.frame(summary(mlr_2025)$coefficients) %>%
  rownames_to_column("Variable") %>%
  filter(Variable != "(Intercept)") %>%
  rename(Estimate = Estimate, SE = `Std. Error`, p_value = `Pr(>|t|)`) %>%
  mutate(
    Significant = ifelse(p_value < 0.05, "Significant (p < 0.05)", "Not Significant"),
    CI_lower    = Estimate - 1.96 * SE,
    CI_upper    = Estimate + 1.96 * SE
  )

ggplot(coef_df, aes(x = reorder(Variable, Estimate), y = Estimate,
                    colour = Significant)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  coord_flip() +
  scale_colour_manual(values = c("Significant (p < 0.05)" = "#E8734A",
                                 "Not Significant"         = "#4A90D9")) +
  labs(title    = "Regression Coefficients with 95% Confidence Intervals — FY2025",
       subtitle = "Dependent variable: Log(Total Property Value)",
       x = NULL, y = "Coefficient Estimate", colour = NULL) +
  theme_minimal(base_size = 12)


# -- 5. Multiple Regression — FY2023 ------------------------------------------
# Identical specification to Section 4 for direct coefficient comparison.

cat("\n── Multiple Regression (8 features) — FY2023 ──\n")

mlr_2023 <- lm(
  as.formula(paste("log_total_value ~ log_land_sf + log_living_area + bed_rms +",
                   "full_bth + hlf_bth +", fp_col_2023, "+ num_parking + age_of_property")),
  data = df2023
)

cat("Adj R² FY2023 =", round(summary(mlr_2023)$adj.r.squared, 4),
    "| Adj R² FY2025 =", round(summary(mlr_2025)$adj.r.squared, 4), "\n")

print(summary(mlr_2023))


# -- 6. Multiple Regression with Building Type Dummies — FY2025 ---------------
# Adding building type as a categorical variable improves Adj R² by
# approximately 12 percentage points (0.536 → 0.656), confirming that
# building style captures meaningful variation unexplained by structural
# features alone.
#
# Reference level: Colonial — the most common building type in Boston
# after prefix stripping. All other type coefficients represent their
# log-scale value premium or discount relative to a Colonial property
# of equivalent size, age, and features.

cat("\n── Multiple Regression with Building Type Dummies — FY2025 ──\n")

top_types <- df2025 %>%
  count(bldg_type_clean, sort = TRUE) %>%
  filter(n >= 200) %>%
  pull(bldg_type_clean)

df2025_typed <- df2025 %>%
  filter(bldg_type_clean %in% top_types) %>%
  mutate(bldg_type_clean = relevel(factor(bldg_type_clean), ref = "Colonial"))

cat("Reference level :", levels(df2025_typed$bldg_type_clean)[1], "\n")
cat("Types included  :", length(top_types), "\n")

mlr_type_2025 <- lm(
  as.formula(paste("log_total_value ~ log_land_sf + log_living_area + bed_rms +",
                   "full_bth +", fp_col_2025, "+ num_parking + age_of_property + bldg_type_clean")),
  data = df2025_typed
)

cat("Adj R² with dummies :", round(summary(mlr_type_2025)$adj.r.squared, 4), "\n")
cat("Adj R² without      :", round(summary(mlr_2025)$adj.r.squared, 4), "\n")

print(summary(mlr_type_2025))

ggplot(df2025_typed %>% filter(bldg_type_clean %in% top_types[1:6]),
       aes(x = log_living_area, y = log_total_value)) +
  geom_point(alpha = 0.2, colour = "#4A90D9", size = 0.7) +
  geom_smooth(method = "lm", colour = "#E8734A", se = FALSE, linewidth = 1) +
  facet_wrap(~ bldg_type_clean, scales = "free") +
  labs(title = "Value vs Living Area by Building Type — FY2025",
       x = "Log(Living Area)", y = "Log(Total Value)") +
  theme_minimal(base_size = 11)


# -- 7. Coefficient Comparison: FY2023 vs FY2025 ------------------------------
# Positive Change = the marginal value of that feature increased.
# Negative Change = the marginal value of that feature decreased.
#
# Key finding: age_of_property coefficient dropped from 0.00137 (FY2023)
# to 0.0001 (FY2025), indicating that older properties lost their relative
# premium, likely because newer condominium construction drove much of
# the FY2025 market appreciation.

cat("\n── Coefficient Comparison: FY2023 vs FY2025 ──\n")

coef_2023 <- as.data.frame(summary(mlr_2023)$coefficients) %>%
  rownames_to_column("Variable") %>%
  select(Variable, Estimate_2023 = Estimate) %>%
  mutate(Variable = gsub("^fire_place", "fireplaces", Variable))

coef_2025_tbl <- as.data.frame(summary(mlr_2025)$coefficients) %>%
  rownames_to_column("Variable") %>%
  select(Variable, Estimate_2025 = Estimate)

coef_comparison <- inner_join(coef_2023, coef_2025_tbl, by = "Variable") %>%
  filter(Variable != "(Intercept)") %>%
  mutate(Change = Estimate_2025 - Estimate_2023)

cat("Matched features:", nrow(coef_comparison), "\n")
cat("Largest increase :", coef_comparison$Variable[which.max(coef_comparison$Change)], "\n")
cat("Largest decrease :", coef_comparison$Variable[which.min(coef_comparison$Change)], "\n\n")

print(coef_comparison)

ggplot(coef_comparison,
       aes(x = reorder(Variable, Change), y = Change,
           fill = ifelse(Change > 0, "Increased", "Decreased"))) +
  geom_col(alpha = 0.85) +
  coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  scale_fill_manual(values = c("Increased" = "#E8734A", "Decreased" = "#4A90D9")) +
  labs(title    = "Change in Regression Coefficients: FY2023 to FY2025",
       subtitle = "Positive values indicate a feature's marginal value increased",
       x = NULL, y = "Coefficient Change (FY2025 − FY2023)", fill = NULL) +
  theme_minimal(base_size = 12)


# -- 8. Residual Diagnostics — FY2025 -----------------------------------------
# Residuals are extracted into a standalone dataframe to avoid a row-count
# mismatch: the model drops 11 observations with missing values during
# fitting, so residuals() returns 133,277 rows while df2025 has 133,288.
# Using fitted() and residuals() directly resolves this cleanly.
#
# A mean residual near zero confirms the model is unbiased on average.
# A fan shape in the residuals-vs-fitted plot would indicate
# heteroscedasticity, which is common in property value data.

cat("\n── Residual Diagnostics — FY2025 ──\n")

residual_df <- data.frame(
  fitted    = fitted(mlr_2025),
  residuals = residuals(mlr_2025)
)

cat("Residual rows:", nrow(residual_df),
    "| mean:", round(mean(residual_df$residuals), 6),
    "| sd:", round(sd(residual_df$residuals), 4), "\n")

ggplot(residual_df, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.12, colour = "#4A90D9", size = 0.6) +
  geom_hline(yintercept = 0, colour = "#E8734A", linewidth = 1) +
  geom_smooth(se = FALSE, colour = "grey50", linewidth = 0.8) +
  labs(title    = "Residuals vs Fitted Values — FY2025 Multiple Regression",
       subtitle = "Random scatter around zero indicates no systematic model misfit",
       x = "Fitted Values (log scale)", y = "Residuals") +
  theme_minimal(base_size = 12)

ggplot(residual_df, aes(x = residuals)) +
  geom_histogram(binwidth = 0.05, fill = "#4A90D9", colour = "white", alpha = 0.85) +
  labs(title    = "Residual Distribution — FY2025 Multiple Regression",
       subtitle = "Approximately normal distribution centred at zero is expected",
       x = "Residuals", y = "Count") +
  theme_minimal(base_size = 12)

cat("\nRegression analysis complete.\n")