# ============================================================================
# Boston Property Assessment — Exploratory Data Analysis (FY2025)
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2025 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : Exploratory analysis of the FY2025 assessment dataset. Mirrors
#           the structure of eda_2023.R to enable direct year-over-year
#           comparison. Results serve as the FY2025 baseline for analyses
#           in comparative_analysis.R and the modelling scripts.
#
# Schema notes (FY2025 vs FY2023):
#   - ZIP code column  : "zip_code"   (FY2023 uses "zipcode")
#   - Kitchens column  : "kitchens"   (FY2023 uses "kitchen")
#   - Fireplaces column: "fireplaces" (FY2023 uses "fire_place")
#   - Heat column      : "heat_system"(FY2023 uses "heat_fuel")
#   - gross_tax        : stored with "$" prefix requiring extra cleaning
#   - sfyi_value       : new column, not present in FY2023
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

housing_data <- read.csv("./data/FY2025_property_assessment.csv") %>%
  clean_names()

cat("Data loaded:", nrow(housing_data), "rows,", ncol(housing_data), "columns\n")


# -- 2. Type Conversion -------------------------------------------------------
# FY2025 introduces additional formatting issues relative to FY2023:
#   - gross_tax carries a leading "$" symbol
#   - sfyi_value is a new numeric field with comma formatting

housing_data <- housing_data %>%
  mutate(
    unit_num    = suppressWarnings(as.numeric(unit_num)),
    land_sf     = as.numeric(gsub(",", "", land_sf)),
    land_value  = as.numeric(gsub(",", "", land_value)),
    bldg_value  = as.numeric(gsub(",", "", bldg_value)),
    sfyi_value  = as.numeric(gsub(",", "", sfyi_value)),
    total_value = as.numeric(gsub(",", "", total_value)),
    gross_tax   = as.numeric(gsub("[^0-9.]", "", gross_tax))
  )

cat("Numeric conversion complete | total_value: $",
    format(min(housing_data$total_value, na.rm = TRUE), big.mark = ","), "–",
    format(max(housing_data$total_value, na.rm = TRUE), big.mark = ","), "\n")
cat("Gross tax range: $",
    format(min(housing_data$gross_tax, na.rm = TRUE), big.mark = ","), "–",
    format(max(housing_data$gross_tax, na.rm = TRUE), big.mark = ","), "\n")


# -- 3. ZIP Code Standardisation ----------------------------------------------

housing_data <- housing_data %>%
  mutate(zip_code = sprintf("%05d", as.numeric(zip_code)))

cat("ZIP codes standardised | sample:", paste(head(housing_data$zip_code, 3), collapse = ", "), "\n")


# -- 4. Condition Label Normalisation -----------------------------------------

housing_data <- housing_data %>%
  mutate(
    overall_cond = case_when(
      overall_cond %in% c("EX - Excellent", "E - Excellent")             ~ "E - Excellent",
      overall_cond %in% c("VG - Very Good",  "G - Good")                 ~ "G - Good",
      overall_cond %in% c("VP - Very Poor",  "P - Poor", "US - Unsound") ~ "P - Poor",
      TRUE ~ overall_cond
    )
  )

cat("Condition labels normalised | unique values:",
    paste(sort(unique(na.omit(housing_data$overall_cond))), collapse = ", "), "\n")


# -- 5. Factor Encoding -------------------------------------------------------
# FY2025 uses heat_system instead of heat_fuel and adds corner_unit.
# All other ordinal and nominal factor definitions match FY2023.

cond_levels         <- c("P - Poor", "F - Fair", "A - Average", "G - Good", "E - Excellent")
bthrm_levels        <- c("N - No Remodeling", "S - Semi-Modern", "M - Modern", "L - Luxury")
kitchen_type_levels <- c("N - None", "O - One Person", "P - Pullman", "F - Full Eat In")
heat_system_levels  <- c("N - None", "C - Common", "I - Indiv. Cntrl", "Y - Self Contained")
orientation_levels  <- c("A - Rear Above", "B - Rear Below", "C - Courtyard", "E - End",
                         "F - Front/Street", "M - Middle", "T - Through")
prop_view_levels    <- c("P - Poor", "F - Fair", "A - Average", "G - Good",
                         "E - Excellent", "S - Special")
corner_unit_levels  <- c("N - No", "Y - Yes")

housing_data <- housing_data %>%
  mutate(
    overall_cond    = factor(overall_cond,  levels = cond_levels,         ordered = TRUE),
    bthrm_style1    = factor(bthrm_style1,  levels = bthrm_levels,        ordered = TRUE),
    kitchen_type    = factor(kitchen_type,  levels = kitchen_type_levels,  ordered = TRUE),
    heat_system     = factor(heat_system,   levels = heat_system_levels,   ordered = TRUE),
    orientation     = factor(orientation,   levels = orientation_levels,   ordered = TRUE),
    prop_view       = factor(prop_view,     levels = prop_view_levels,     ordered = TRUE),
    corner_unit     = factor(corner_unit,   levels = corner_unit_levels,   ordered = TRUE),
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
# FY2025 uses "kitchens" and "fireplaces" — FY2023 uses "kitchen" and "fire_place".

numeric_cols          <- c("land_sf", "total_value", "land_value", "bldg_value",
                           "gross_tax", "bed_rms", "full_bth", "hlf_bth",
                           "kitchens", "tt_rms", "yr_built", "fireplaces")
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
ggplot(housing_data, aes(x = log10(total_value))) +
  geom_histogram(binwidth = 0.1, fill = "#4A90D9", alpha = 0.85, colour = "white") +
  labs(title    = "Distribution of Property Values — FY2025",
       subtitle = "Log10 scale applied to address right skew",
       x = "Log\u2081\u2080(Total Value)", y = "Count") +
  theme_minimal(base_size = 12)

# 9.2 Property Value vs Land Area
ggplot(housing_data, aes(x = land_sf, y = total_value)) +
  geom_point(alpha = 0.25, colour = "#4A90D9", size = 0.7) +
  geom_smooth(method = "lm", colour = "#E8734A", se = FALSE, linewidth = 1) +
  scale_x_log10() + scale_y_log10() +
  labs(title    = "Property Value vs Land Area — FY2025",
       subtitle = "Log–log scale; compare slope with FY2023 to detect land premium shift",
       x = "Land Size (sq ft, log scale)", y = "Total Value (log scale)") +
  theme_minimal(base_size = 12)

# 9.3 Gross Tax vs Property Value
ggplot(housing_data, aes(x = total_value, y = gross_tax)) +
  geom_point(alpha = 0.25, colour = "#E8734A", size = 0.7) +
  scale_x_log10() + scale_y_log10() +
  labs(title    = "Gross Tax vs Property Value — FY2025",
       subtitle = "A weak relationship suggests inconsistent tax burden across properties",
       x = "Total Value (log scale)", y = "Gross Tax (log scale)") +
  theme_minimal(base_size = 12)

# 9.4 Property Condition Distribution
ggplot(housing_data %>% filter(!is.na(overall_cond)), aes(x = overall_cond)) +
  geom_bar(fill = "#4A90D9", alpha = 0.85) +
  labs(title = "Property Condition Distribution — FY2025",
       x = "Overall Condition", y = "Count") +
  theme_minimal(base_size = 12)

# 9.5 Bedrooms vs Bathrooms
ggplot(housing_data, aes(x = bed_rms, y = full_bth)) +
  geom_point(alpha = 0.35, colour = "#4A90D9", size = 0.8) +
  labs(title = "Bedrooms vs Full Bathrooms — FY2025",
       x = "Number of Bedrooms", y = "Number of Full Bathrooms") +
  theme_minimal(base_size = 12)

# 9.6 Average Value by ZIP Code (Top 20)
zip_value <- housing_data %>%
  group_by(zip_code) %>%
  summarise(avg_value = mean(total_value, na.rm = TRUE), n = n(), .groups = "drop") %>%
  filter(n >= 50) %>%
  arrange(desc(avg_value)) %>%
  slice_head(n = 20)

cat("\nTop ZIP code by average value:", zip_value$zip_code[1],
    "($", format(round(zip_value$avg_value[1]), big.mark = ","), ")\n")

ggplot(zip_value, aes(x = reorder(zip_code, avg_value), y = avg_value / 1e6)) +
  geom_col(fill = "#4A90D9", alpha = 0.85) +
  coord_flip() +
  labs(title    = "Top 20 ZIP Codes by Average Property Value — FY2025",
       x = "ZIP Code", y = "Average Value ($M)") +
  theme_minimal(base_size = 12)

# 9.7 Building Type Distribution (Top 10)
bldg_type_count <- housing_data %>%
  mutate(bldg_type_clean = sub("^.* - ", "", as.character(bldg_type))) %>%
  count(bldg_type_clean, sort = TRUE) %>%
  slice_head(n = 10)

cat("Most common building type:", bldg_type_count$bldg_type_clean[1],
    "(", bldg_type_count$n[1], "properties)\n")

ggplot(bldg_type_count, aes(x = reorder(bldg_type_clean, n), y = n)) +
  geom_col(fill = "#E8734A", alpha = 0.85) +
  coord_flip() +
  labs(title = "Top 10 Building Types by Count — FY2025",
       x = "Building Type", y = "Count") +
  theme_minimal(base_size = 12)

# 9.8 Correlation Heatmap
# FY2025 uses "fireplaces" — FY2023 uses "fire_place".
numeric_data <- housing_data %>%
  select(total_value, land_value, bldg_value, gross_tax,
         land_sf, living_area, bed_rms, full_bth, fireplaces, num_parking) %>%
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

cat("\nEDA FY2025 complete.\n")
