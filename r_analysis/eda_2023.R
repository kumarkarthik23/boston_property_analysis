# ============================================================================
# Boston Property Assessment — Exploratory Data Analysis (FY2023)
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2023 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : Exploratory analysis of the FY2023 assessment dataset covering
#           data quality, descriptive statistics, outlier detection, and
#           eight diagnostic visualisations. Results serve as the FY2023
#           baseline for year-over-year comparison in comparative_analysis.R.
#
# Schema notes (FY2023 vs FY2025):
#   - ZIP code column  : "zipcode"    (FY2025 uses "zip_code")
#   - Kitchens column  : "kitchen"    (FY2025 uses "kitchens")
#   - Fireplaces column: "fire_place" (FY2025 uses "fireplaces")
#   - Heat column      : "heat_fuel"  (FY2025 uses "heat_system")
# ============================================================================


# -- Environment --------------------------------------------------------------

rm(list = ls())
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)
options(scipen = 100)
cat("\014")

library(tidyverse)
library(janitor)
library(ggplot2)
library(corrplot)


# -- 1. Data Loading ----------------------------------------------------------

housing_data <- read.csv("./data/FY2023_property_assessment.csv") %>%
  clean_names()

cat("Data loaded:", nrow(housing_data), "rows,", ncol(housing_data), "columns\n")


# -- 2. Type Conversion -------------------------------------------------------
# Several numeric fields are stored as formatted strings in the raw CSV
# (e.g. land_sf = "1,150", gross_tax = "$8,422"). Commas and currency
# symbols are stripped before casting to numeric.

housing_data <- housing_data %>%
  mutate(
    land_sf     = as.numeric(gsub(",", "", land_sf)),
    land_value  = as.numeric(gsub(",", "", land_value)),
    bldg_value  = as.numeric(gsub(",", "", bldg_value)),
    total_value = as.numeric(gsub(",", "", total_value)),
    gross_tax   = as.numeric(gsub("[^0-9.]", "", gross_tax)),
    gross_area  = as.numeric(gsub(",", "", gross_area)),
    living_area = as.numeric(gsub(",", "", living_area))
  )

cat("Numeric conversion complete | total_value: $",
    format(min(housing_data$total_value, na.rm = TRUE), big.mark = ","), "–",
    format(max(housing_data$total_value, na.rm = TRUE), big.mark = ","), "\n")


# -- 3. ZIP Code Standardisation ----------------------------------------------
# Zero-pad all ZIP codes to five digits to prevent mismatches
# between values like "2128" and "02128".

housing_data <- housing_data %>%
  mutate(zipcode = sprintf("%05d", as.numeric(zipcode)))

cat("ZIP codes standardised | sample:", paste(head(housing_data$zipcode, 3), collapse = ", "), "\n")


# -- 4. Condition Label Normalisation -----------------------------------------
# The overall_cond field uses inconsistent codes across records
# (e.g. "EX - Excellent" and "E - Excellent" represent the same category).
# All variants are mapped to canonical labels before factor encoding.

housing_data <- housing_data %>%
  mutate(
    overall_cond = case_when(
      overall_cond %in% c("EX - Excellent", "E - Excellent")            ~ "E - Excellent",
      overall_cond %in% c("VG - Very Good",  "G - Good")                ~ "G - Good",
      overall_cond %in% c("AVG - Default - Average", "A - Average")     ~ "A - Average",
      overall_cond %in% c("VP - Very Poor",  "P - Poor", "US - Unsound")~ "P - Poor",
      TRUE ~ overall_cond
    )
  )

cat("Condition labels normalised | unique values:",
    paste(sort(unique(na.omit(housing_data$overall_cond))), collapse = ", "), "\n")


# -- 5. Factor Encoding -------------------------------------------------------
# Ordinal variables are encoded as ordered factors so that their natural
# ranking is preserved during model encoding and visualisation ordering.
# Nominal variables are encoded as unordered factors.

cond_levels         <- c("P - Poor", "F - Fair", "A - Average", "G - Good", "E - Excellent")
bthrm_levels        <- c("N - No Remodeling", "S - Semi-Modern", "M - Modern", "L - Luxury")
kitchen_type_levels <- c("N - None", "O - One Person", "P - Pullman", "F - Full Eat In")
heat_fuel_levels    <- c("N - None", "G - Gas", "O - Oil", "E - Electric", "W - Wood")
orientation_levels  <- c("A - Rear Above", "B - Rear Below", "C - Courtyard", "E - End",
                         "F - Front/Street", "M - Middle", "T - Through")
prop_view_levels    <- c("P - Poor", "F - Fair", "A - Average", "G - Good",
                         "E - Excellent", "S - Special")

housing_data <- housing_data %>%
  mutate(
    overall_cond    = factor(overall_cond, levels = cond_levels,          ordered = TRUE),
    bthrm_style1    = factor(bthrm_style1, levels = bthrm_levels,         ordered = TRUE),
    kitchen_type    = factor(kitchen_type, levels = kitchen_type_levels,   ordered = TRUE),
    heat_fuel       = factor(heat_fuel,    levels = heat_fuel_levels,      ordered = FALSE),
    orientation     = factor(orientation,  levels = orientation_levels,    ordered = TRUE),
    prop_view       = factor(prop_view,    levels = prop_view_levels,      ordered = TRUE),
    lu_desc         = as.factor(lu_desc),
    bldg_type       = as.factor(bldg_type),
    own_occ         = as.factor(own_occ),
    structure_class = as.factor(structure_class)
  )

cat("Factor encoding complete | overall_cond levels:",
    paste(levels(housing_data$overall_cond), collapse = " < "), "\n")


# -- 6. Data Quality Assessment -----------------------------------------------

missing_values  <- colSums(is.na(housing_data))
missing_nonzero <- missing_values[missing_values > 0]
missing_pct     <- colSums(is.na(housing_data)) / nrow(housing_data) * 100
num_duplicates  <- sum(duplicated(housing_data))

cat("\nMissing values | columns with NAs:", length(missing_nonzero), "\n")
if (length(missing_nonzero) > 0)
  print(round(sort(missing_pct[missing_pct > 0], decreasing = TRUE), 2))

cat("Duplicate rows:", num_duplicates, "\n")


# -- 7. Outlier Detection -----------------------------------------------------
# Values outside the Tukey fences (Q1 − 1.5·IQR, Q3 + 1.5·IQR) are flagged.
# Outlier counts inform downstream decisions about filtering and imputation.
# Note: FY2023 uses "kitchen" and "fire_place" rather than FY2025 naming.

numeric_cols          <- c("land_sf", "total_value", "land_value", "bldg_value",
                           "gross_tax", "bed_rms", "full_bth", "hlf_bth",
                           "kitchen", "tt_rms", "yr_built", "fire_place")
existing_numeric_cols <- intersect(numeric_cols, names(housing_data))
missing_cols          <- setdiff(numeric_cols, names(housing_data))

if (length(missing_cols) > 0)
  cat("Columns not found (skipped):", paste(missing_cols, collapse = ", "), "\n")

detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  x < (q1 - 1.5 * (q3 - q1)) | x > (q3 + 1.5 * (q3 - q1))
}

outlier_counts <- sapply(housing_data[existing_numeric_cols],
                         function(col) sum(detect_outliers(col), na.rm = TRUE))

cat("\nOutlier counts per column:\n")
print(outlier_counts)
cat("Column with most outliers:", names(which.max(outlier_counts)),
    "(", max(outlier_counts), ")\n")


# -- 8. Summary Statistics ----------------------------------------------------

cat("\nDescriptive Statistics — Key Numeric Columns\n")
print(summary(housing_data %>%
  select(total_value, land_value, bldg_value, gross_tax,
         land_sf, living_area, bed_rms, full_bth, yr_built)))

cat("\nMedian total value : $", format(round(median(housing_data$total_value, na.rm = TRUE)),
                                       big.mark = ","), "\n")
cat("Median gross tax   : $", format(round(median(housing_data$gross_tax,   na.rm = TRUE)),
                                       big.mark = ","), "\n")


# -- 9. Visualisations --------------------------------------------------------

# 9.1 Property Value Distribution
# Log10 scale is applied because property values are right-skewed.
ggplot(housing_data, aes(x = log10(total_value))) +
  geom_histogram(binwidth = 0.1, fill = "#4A90D9", alpha = 0.85, colour = "white") +
  labs(title    = "Distribution of Property Values — FY2023",
       subtitle = "Log10 scale applied to address right skew",
       x = "Log\u2081\u2080(Total Value)", y = "Count") +
  theme_minimal(base_size = 12)

# 9.2 Property Value vs Land Area
# Both axes log-transformed. Regression line shows the land–value relationship.
ggplot(housing_data, aes(x = land_sf, y = total_value)) +
  geom_point(alpha = 0.25, colour = "#4A90D9", size = 0.7) +
  geom_smooth(method = "lm", colour = "#E8734A", se = FALSE, linewidth = 1) +
  scale_x_log10() + scale_y_log10() +
  labs(title    = "Property Value vs Land Area — FY2023",
       subtitle = "Log–log scale; slope = land-size elasticity of value",
       x = "Land Size (sq ft, log scale)", y = "Total Value (log scale)") +
  theme_minimal(base_size = 12)

# 9.3 Gross Tax vs Property Value
# Weak correlation indicates taxation inequities across the portfolio.
ggplot(housing_data, aes(x = total_value, y = gross_tax)) +
  geom_point(alpha = 0.25, colour = "#E8734A", size = 0.7) +
  scale_x_log10() + scale_y_log10() +
  labs(title    = "Gross Tax vs Property Value — FY2023",
       subtitle = "A weak relationship suggests inconsistent tax burden across properties",
       x = "Total Value (log scale)", y = "Gross Tax (log scale)") +
  theme_minimal(base_size = 12)

# 9.4 Property Condition Distribution
ggplot(housing_data %>% filter(!is.na(overall_cond)), aes(x = overall_cond)) +
  geom_bar(fill = "#4A90D9", alpha = 0.85) +
  labs(title = "Property Condition Distribution — FY2023",
       x = "Overall Condition", y = "Count") +
  theme_minimal(base_size = 12)

# 9.5 Bedrooms vs Bathrooms
# Unusual combinations may indicate multi-family dwellings or data errors.
ggplot(housing_data, aes(x = bed_rms, y = full_bth)) +
  geom_point(alpha = 0.35, colour = "#4A90D9", size = 0.8) +
  labs(title = "Bedrooms vs Full Bathrooms — FY2023",
       x = "Number of Bedrooms", y = "Number of Full Bathrooms") +
  theme_minimal(base_size = 12)

# 9.6 Average Value by ZIP Code (Top 20)
# Restricted to ZIP codes with ≥ 50 properties to suppress small-sample noise.
zip_value <- housing_data %>%
  group_by(zipcode) %>%
  summarise(avg_value = mean(total_value, na.rm = TRUE), n = n(), .groups = "drop") %>%
  filter(n >= 50) %>%
  arrange(desc(avg_value)) %>%
  slice_head(n = 20)

cat("\nTop ZIP code by average value:", zip_value$zipcode[1],
    "($", format(round(zip_value$avg_value[1]), big.mark = ","), ")\n")

ggplot(zip_value, aes(x = reorder(zipcode, avg_value), y = avg_value / 1e6)) +
  geom_col(fill = "#4A90D9", alpha = 0.85) +
  coord_flip() +
  labs(title    = "Top 20 ZIP Codes by Average Property Value — FY2023",
       x = "ZIP Code", y = "Average Value ($M)") +
  theme_minimal(base_size = 12)

# 9.7 Building Type Distribution (Top 10)
# Code prefixes (e.g. "RE - Row End") are stripped for legibility.
bldg_type_count <- housing_data %>%
  mutate(bldg_type_clean = sub("^.* - ", "", as.character(bldg_type))) %>%
  count(bldg_type_clean, sort = TRUE) %>%
  slice_head(n = 10)

cat("Most common building type:", bldg_type_count$bldg_type_clean[1],
    "(", bldg_type_count$n[1], "properties)\n")

ggplot(bldg_type_count, aes(x = reorder(bldg_type_clean, n), y = n)) +
  geom_col(fill = "#E8734A", alpha = 0.85) +
  coord_flip() +
  labs(title = "Top 10 Building Types by Count — FY2023",
       x = "Building Type", y = "Count") +
  theme_minimal(base_size = 12)

# 9.8 Correlation Heatmap
# Computed on complete cases only. Strong pairwise correlations (|r| > 0.8)
# signal multicollinearity to address before regression modelling.
# Note: FY2023 uses "fire_place" — FY2025 uses "fireplaces".
numeric_data <- housing_data %>%
  select(total_value, land_value, bldg_value, gross_tax,
         land_sf, living_area, bed_rms, full_bth, fire_place, num_parking) %>%
  na.omit()

cat("Correlation matrix | rows used:", nrow(numeric_data), "\n")

cor_matrix <- cor(numeric_data)

cat("Strongest predictor of total_value:",
    names(which.max(cor_matrix["total_value",
                               names(cor_matrix["total_value",]) != "total_value"])), "\n")

corrplot(cor_matrix,
         method      = "color",
         type        = "upper",
         tl.col      = "black",
         tl.srt      = 45,
         addCoef.col = "black",
         number.cex  = 0.7,
         col         = colorRampPalette(c("#4A90D9", "white", "#E8734A"))(200))

cat("\nEDA FY2023 complete.\n")

