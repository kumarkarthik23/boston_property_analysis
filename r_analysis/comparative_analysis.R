# ============================================================================
# Boston Property Assessment — Comparative Analysis FY2023 vs FY2025
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2023 and FY2025 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : Year-over-year comparative analysis addressing five research
#           questions using both cross-sectional aggregations and matched
#           property records (same PID across both years).
#
# Research Questions:
#   RQ1 — How have property values changed between 2023 and 2025?
#   RQ2 — Has the gross tax burden shifted relative to assessed value?
#   RQ3 — Has the bedroom and bathroom price premium changed?
#   RQ4 — Has the distribution of building types changed?
#   RQ5 — Has the market premium for parking spaces shifted?
#
# Confirmed results:
#   Matched properties (n = 63,667):
#     Median value change : +17.34%
#     Median tax change   : +24.88%  (taxes rose 44% faster than values)
#     Median land change  : +18.63%
#     Median bldg change  : +15.57%
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


# -- 1. Shared Cleaning Function ----------------------------------------------
# A single function applied to both datasets guarantees identical
# preprocessing so that observed year-over-year differences reflect real
# market changes rather than pipeline inconsistencies.
#
# Design decisions:
#   across(everything(), as.character) — flattens mismatched column types
#     between FY2023 (st_num = character) and FY2025 (st_num = integer)
#     before bind_rows() is called; prevents type-conflict errors.
#   IQR outlier removal — focuses the comparison on typical residential
#     properties by capping extreme luxury outliers in total_value.
#   Targeted NA filter — na.omit() on the full dataframe drops all rows
#     because sparse columns (kitchen_style, heat_system, orientation)
#     are NA for almost every record. We filter only on the columns
#     the analysis actually uses.

clean_dataset <- function(df, year) {
  df %>%
    mutate(across(everything(), as.character)) %>%
    mutate(
      land_sf     = as.numeric(gsub(",",       "", land_sf)),
      gross_area  = as.numeric(gsub(",",       "", gross_area)),
      living_area = as.numeric(gsub(",",       "", living_area)),
      land_value  = as.numeric(gsub(",",       "", land_value)),
      bldg_value  = as.numeric(gsub(",",       "", bldg_value)),
      total_value = as.numeric(gsub(",",       "", total_value)),
      gross_tax   = as.numeric(gsub("[^0-9.]", "", gross_tax)),
      yr_built    = as.numeric(yr_built),
      bed_rms     = as.numeric(bed_rms),
      full_bth    = as.numeric(full_bth),
      hlf_bth     = as.numeric(hlf_bth),
      num_parking = as.numeric(num_parking)
    ) %>%
    filter(
      total_value > 0, gross_tax   > 0,
      living_area > 0, land_sf     > 0,
      !is.na(yr_built), yr_built > 1700,
      yr_built <= as.numeric(format(Sys.Date(), "%Y"))
    ) %>%
    {
      q1 <- quantile(.$total_value, 0.25, na.rm = TRUE)
      q3 <- quantile(.$total_value, 0.75, na.rm = TRUE)
      filter(., total_value >= q1 - 1.5 * (q3 - q1),
                total_value <= q3 + 1.5 * (q3 - q1))
    } %>%
    mutate(
      fiscal_year        = year,
      price_per_sqft     = total_value / living_area,
      age_of_property    = year - yr_built,
      tax_to_value_ratio = gross_tax / total_value * 100,
      bldg_type_clean    = sub("^.* - ", "", bldg_type),
      city_clean         = str_to_title(trimws(city))
    ) %>%
    filter(
      !is.na(total_value), !is.na(gross_tax),   !is.na(land_sf),
      !is.na(living_area), !is.na(land_value),  !is.na(bldg_value),
      !is.na(bldg_type),   !is.na(city),         !is.na(yr_built),
      !is.na(price_per_sqft), !is.na(tax_to_value_ratio)
    )
}


# -- 2. Load and Clean --------------------------------------------------------

cat("Loading FY2023...\n")
df2023 <- read.csv("./data/FY2023_property_assessment.csv") %>%
  clean_names() %>% clean_dataset(2023)
cat("FY2023 ready:", nrow(df2023), "records\n")

cat("Loading FY2025...\n")
df2025 <- read.csv("./data/FY2025_property_assessment.csv") %>%
  clean_names() %>% clean_dataset(2025)
cat("FY2025 ready:", nrow(df2025), "records\n")


# -- 3. Matched Property Analysis (Same PID) ----------------------------------
# Matching on the unique property identifier (PID) ensures a like-for-like
# comparison on the same physical properties across both years, isolating
# genuine appreciation from changes in the portfolio composition.

matched_pids <- intersect(df2023$pid, df2025$pid)
cat("\nMatched properties (same PID in both years):", length(matched_pids), "\n")

yoy <- df2023 %>%
  filter(pid %in% matched_pids) %>%
  select(pid, total_value, gross_tax, land_value, bldg_value,
         living_area, price_per_sqft, tax_to_value_ratio, bldg_type_clean) %>%
  inner_join(
    df2025 %>%
      filter(pid %in% matched_pids) %>%
      select(pid, total_value, gross_tax, land_value, bldg_value,
             living_area, price_per_sqft, tax_to_value_ratio, bldg_type_clean),
    by = "pid", suffix = c("_2023", "_2025")
  ) %>%
  mutate(
    value_change_pct = (total_value_2025 - total_value_2023) / total_value_2023 * 100,
    tax_change_pct   = (gross_tax_2025   - gross_tax_2023)   / gross_tax_2023   * 100,
    land_change_pct  = (land_value_2025  - land_value_2023)  / land_value_2023  * 100,
    bldg_change_pct  = (bldg_value_2025  - bldg_value_2023)  / bldg_value_2023  * 100
  )

cat("\nMedian year-over-year changes (matched properties):\n")
cat("  Property value : +", round(median(yoy$value_change_pct, na.rm = TRUE), 2), "%\n")
cat("  Gross tax      : +", round(median(yoy$tax_change_pct,   na.rm = TRUE), 2), "%\n")
cat("  Land value     : +", round(median(yoy$land_change_pct,  na.rm = TRUE), 2), "%\n")
cat("  Building value : +", round(median(yoy$bldg_change_pct,  na.rm = TRUE), 2), "%\n")


# -- 4. Combined Dataset for Cross-sectional Analysis ------------------------

combined <- bind_rows(df2023, df2025) %>%
  mutate(fiscal_year = as.factor(fiscal_year))

cat("\nCombined dataset:", nrow(combined), "records |",
    "FY2023:", sum(combined$fiscal_year == 2023), "|",
    "FY2025:", sum(combined$fiscal_year == 2025), "\n")

top_types <- df2025 %>%
  count(bldg_type_clean, sort = TRUE) %>%
  slice_head(n = 8) %>%
  pull(bldg_type_clean)


# -- 5. RQ1 — Property Value Appreciation -------------------------------------

# Value distribution overlay
ggplot(combined, aes(x = log10(total_value), fill = fiscal_year)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  labs(title    = "Property Value Distribution: FY2023 vs FY2025",
       subtitle = "Rightward shift in 2025 indicates broad market appreciation",
       x = "Log\u2081\u2080(Total Property Value)", y = "Density", fill = "Year") +
  theme_minimal(base_size = 12)

# Median value by building type
median_by_type <- combined %>%
  group_by(fiscal_year, bldg_type_clean) %>%
  summarise(median_value = median(total_value, na.rm = TRUE), .groups = "drop") %>%
  filter(bldg_type_clean %in% top_types)

ggplot(median_by_type,
       aes(x = reorder(bldg_type_clean, median_value),
           y = median_value / 1e6, fill = fiscal_year)) +
  geom_col(position = "dodge", alpha = 0.85) +
  coord_flip() +
  scale_fill_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  scale_y_continuous(labels = dollar_format(suffix = "M")) +
  labs(title = "Median Property Value by Building Type: FY2023 vs FY2025",
       x = "Building Type", y = "Median Total Value", fill = "Year") +
  theme_minimal(base_size = 12)

# YoY value change distribution (matched properties)
ggplot(yoy, aes(x = value_change_pct)) +
  geom_histogram(binwidth = 2, fill = "#4A90D9", colour = "white", alpha = 0.85) +
  geom_vline(xintercept = median(yoy$value_change_pct, na.rm = TRUE),
             colour = "#E8734A", linewidth = 1, linetype = "dashed") +
  labs(title    = "Year-over-Year Property Value Change: FY2023 to FY2025",
       subtitle = paste0("Dashed line = median: +",
                         round(median(yoy$value_change_pct, na.rm = TRUE), 1), "%"),
       x = "Value Change (%)", y = "Number of Properties") +
  theme_minimal(base_size = 12)


# -- 6. RQ2 — Taxation Fairness -----------------------------------------------

# Tax-to-value ratio overlay
ggplot(combined %>% filter(tax_to_value_ratio < 5),
       aes(x = tax_to_value_ratio, fill = fiscal_year)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  labs(title    = "Effective Tax Burden: FY2023 vs FY2025",
       subtitle = "Tax-to-value ratio; a rightward shift indicates a higher tax burden in 2025",
       x = "Gross Tax / Total Value (%)", y = "Density", fill = "Year") +
  theme_minimal(base_size = 12)

# Tax change vs value change (matched properties)
ggplot(yoy %>% filter(abs(value_change_pct) < 50, abs(tax_change_pct) < 50),
       aes(x = value_change_pct, y = tax_change_pct)) +
  geom_point(alpha = 0.12, colour = "#4A90D9", size = 0.8) +
  geom_smooth(method = "lm", colour = "#E8734A", se = FALSE, linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  labs(title    = "Property Value Change vs Tax Change: FY2023 to FY2025",
       subtitle = "Points above the diagonal are taxed more than their appreciation warrants",
       x = "Value Change (%)", y = "Tax Change (%)") +
  theme_minimal(base_size = 12)


# -- 7. RQ3 — Bedroom and Bathroom Premium ------------------------------------

bedroom_premium <- combined %>%
  filter(bed_rms >= 1, bed_rms <= 6) %>%
  group_by(fiscal_year, bed_rms) %>%
  summarise(median_value = median(total_value, na.rm = TRUE), .groups = "drop")

ggplot(bedroom_premium,
       aes(x = bed_rms, y = median_value / 1e6,
           colour = fiscal_year, group = fiscal_year)) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  scale_colour_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  scale_y_continuous(labels = dollar_format(suffix = "M")) +
  labs(title  = "Median Value by Number of Bedrooms: FY2023 vs FY2025",
       x = "Number of Bedrooms", y = "Median Total Value", colour = "Year") +
  theme_minimal(base_size = 12)

bathroom_premium <- combined %>%
  filter(full_bth >= 1, full_bth <= 5) %>%
  group_by(fiscal_year, full_bth) %>%
  summarise(median_value = median(total_value, na.rm = TRUE), .groups = "drop")

ggplot(bathroom_premium,
       aes(x = full_bth, y = median_value / 1e6,
           colour = fiscal_year, group = fiscal_year)) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  scale_colour_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  scale_y_continuous(labels = dollar_format(suffix = "M")) +
  labs(title  = "Median Value by Number of Full Bathrooms: FY2023 vs FY2025",
       x = "Number of Full Bathrooms", y = "Median Total Value", colour = "Year") +
  theme_minimal(base_size = 12)


# -- 8. RQ4 — Building Type Distribution --------------------------------------

type_dist <- combined %>%
  filter(bldg_type_clean %in% top_types) %>%
  group_by(fiscal_year, bldg_type_clean) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(fiscal_year) %>%
  mutate(pct = count / sum(count) * 100) %>%
  ungroup()

ggplot(type_dist,
       aes(x = reorder(bldg_type_clean, pct), y = pct, fill = fiscal_year)) +
  geom_col(position = "dodge", alpha = 0.85) +
  coord_flip() +
  scale_fill_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  labs(title = "Building Type Portfolio Share: FY2023 vs FY2025",
       x = "Building Type", y = "Share of Total Properties (%)", fill = "Year") +
  theme_minimal(base_size = 12)


# -- 9. RQ5 — Parking Amenity Premium -----------------------------------------

parking_premium <- combined %>%
  filter(num_parking >= 0, num_parking <= 5) %>%
  group_by(fiscal_year, num_parking) %>%
  summarise(median_value = median(total_value, na.rm = TRUE), .groups = "drop")

ggplot(parking_premium,
       aes(x = num_parking, y = median_value / 1e6,
           colour = fiscal_year, group = fiscal_year)) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  scale_colour_manual(values = c("2023" = "#4A90D9", "2025" = "#E8734A")) +
  scale_y_continuous(labels = dollar_format(suffix = "M")) +
  labs(title  = "Median Value by Number of Parking Spaces: FY2023 vs FY2025",
       x = "Number of Parking Spaces", y = "Median Total Value", colour = "Year") +
  theme_minimal(base_size = 12)


# -- 10. Summary Table --------------------------------------------------------

pct_change <- function(old, new) round((new - old) / old * 100, 2)

summary_tbl <- bind_rows(
  df2023 %>% summarise(year = 2023, n = n(),
    med_value = median(total_value), med_tax    = median(gross_tax),
    med_psf   = median(price_per_sqft), med_ratio = median(tax_to_value_ratio)),
  df2025 %>% summarise(year = 2025, n = n(),
    med_value = median(total_value), med_tax    = median(gross_tax),
    med_psf   = median(price_per_sqft), med_ratio = median(tax_to_value_ratio))
)

cat("\n── Cross-sectional Summary ──\n")
print(summary_tbl)

cat("\n── Year-over-Year Changes (Cross-sectional) ──\n")
cat("  Median property value : ", pct_change(summary_tbl$med_value[1],
                                              summary_tbl$med_value[2]), "%\n")
cat("  Median gross tax      : ", pct_change(summary_tbl$med_tax[1],
                                              summary_tbl$med_tax[2]),   "%\n")
cat("  Median price per sqft : ", pct_change(summary_tbl$med_psf[1],
                                              summary_tbl$med_psf[2]),   "%\n")
cat("  Median tax-to-value   : ", pct_change(summary_tbl$med_ratio[1],
                                              summary_tbl$med_ratio[2]), "pp\n")

cat("\nComparative analysis complete.\n")

