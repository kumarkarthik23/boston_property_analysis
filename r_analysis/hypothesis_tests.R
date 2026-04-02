# ============================================================================
# Boston Property Assessment — Statistical Hypothesis Testing
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2023 and FY2025 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : Formal hypothesis tests to determine whether observed differences
#           between FY2023 and FY2025 are statistically significant.
#
# Tests conducted (α = 0.05 throughout):
#   H1  — Welch two-sample t-test  : Did total property values change?
#   H2  — One-tailed t-test        : Did gross tax increase?
#   H3  — One-tailed t-test        : Do 3+ bedroom properties command a
#                                    significantly higher value than < 3?
#   H4  — Welch two-sample t-test  : Do owner-occupied properties differ
#                                    in assessed value from investor-owned?
#   H5  — One-tailed t-test        : Has the tax-to-value ratio increased?
#   ANOVA + Tukey HSD              : Does building type affect value?
#
# Confirmed results:
#   H1 Reject H0 — values changed significantly (p = 0.000003)
#   H2 Fail to reject — composition effect; see H5 and medians
#   H3 Reject H0 — 3+ bedrooms command higher values (p ≈ 0)
#   H4 Reject H0 — owner vs non-owner differ (p ≈ 0)
#   H5 Reject H0 — tax burden increased (p ≈ 0; ratio 1.088% → 1.158%)
#   ANOVA: F = 510.99, p ≈ 0 (31 building types)
# ============================================================================


# -- Environment --------------------------------------------------------------

rm(list = ls())
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)
options(scipen = 100)
cat("\014")

library(tidyverse)
library(janitor)


# -- 1. Shared Data Loading Function ------------------------------------------
# Identical preprocessing applied to both years ensures observed differences
# are real rather than artefacts of inconsistent cleaning.
#
# price_per_sqft is capped at $5,000/sqft to remove extreme outliers caused
# by properties with near-zero living_area (data entry errors in FY2023).
# Without this cap, the FY2023 mean reaches $12,712/sqft — orders of
# magnitude above a realistic Boston market rate.

load_and_clean <- function(filepath) {
  read.csv(filepath) %>%
    clean_names() %>%
    mutate(
      land_sf     = as.numeric(gsub(",",       "", land_sf)),
      land_value  = as.numeric(gsub(",",       "", land_value)),
      bldg_value  = as.numeric(gsub(",",       "", bldg_value)),
      total_value = as.numeric(gsub(",",       "", total_value)),
      gross_tax   = as.numeric(gsub("[^0-9.]", "", gross_tax)),
      living_area = as.numeric(gsub(",",       "", living_area))
    ) %>%
    filter(
      total_value > 0, gross_tax   > 0,
      living_area > 0, land_sf     > 0,
      yr_built    > 1700,
      yr_built    <= as.numeric(format(Sys.Date(), "%Y"))
    ) %>%
    mutate(
      price_per_sqft     = total_value / living_area,
      tax_to_value_ratio = gross_tax / total_value * 100,
      bldg_type_clean    = sub("^.* - ", "", bldg_type)
    ) %>%
    filter(price_per_sqft < 5000) %>%
    filter(
      !is.na(total_value), !is.na(gross_tax),
      !is.na(living_area), !is.na(land_sf),
      !is.na(price_per_sqft), !is.na(tax_to_value_ratio),
      !is.na(bldg_type), !is.na(own_occ)
    )
}

df2023 <- load_and_clean("./data/FY2023_property_assessment.csv")
df2025 <- load_and_clean("./data/FY2025_property_assessment.csv")

cat("FY2023:", nrow(df2023), "records | FY2025:", nrow(df2025), "records\n")
cat("Price/sqft check | FY2023: $", round(mean(df2023$price_per_sqft), 2),
    "| FY2025: $", round(mean(df2025$price_per_sqft), 2), "\n")


# -- 2. H1 — Did Property Values Change? --------------------------------------
# H0: μ(total_value, 2023) = μ(total_value, 2025)
# H1: μ(total_value, 2023) ≠ μ(total_value, 2025)   [two-tailed]
# Test: Welch t-test (unequal variance assumed given different sample sizes)
#
# Note: the cross-sectional mean appears lower in FY2025 than FY2023 due
# to portfolio composition differences. The test still correctly detects
# a statistically significant shift in the distribution.

cat("\n── H1: Did property values change between 2023 and 2025? ──\n")

t1 <- t.test(df2023$total_value, df2025$total_value)

cat("  Mean FY2023 : $", format(round(mean(df2023$total_value)), big.mark = ","), "\n")
cat("  Mean FY2025 : $", format(round(mean(df2025$total_value)), big.mark = ","), "\n")
cat("  p-value     :", round(t1$p.value, 6), "\n")
cat("  Decision    :", ifelse(t1$p.value < 0.05,
    "Reject H0 — significant change in property values (p < 0.05)",
    "Fail to reject H0"), "\n")


# -- 3. H2 — Did Gross Tax Increase? ------------------------------------------
# H0: μ(gross_tax, 2023) ≥ μ(gross_tax, 2025)
# H1: μ(gross_tax, 2023) < μ(gross_tax, 2025)   [one-tailed, directional]
#
# Mean-based test fails here due to composition effects (FY2023 contains
# proportionally more large commercial properties). The median tells a
# clearer story and H5 provides the most reliable measure of tax burden.

cat("\n── H2: Did gross tax increase between 2023 and 2025? ──\n")

t2 <- t.test(df2023$gross_tax, df2025$gross_tax, alternative = "less")

cat("  Mean   FY2023 : $", format(round(mean(df2023$gross_tax)),   big.mark = ","), "\n")
cat("  Mean   FY2025 : $", format(round(mean(df2025$gross_tax)),   big.mark = ","), "\n")
cat("  Median FY2023 : $", format(round(median(df2023$gross_tax)), big.mark = ","), "\n")
cat("  Median FY2025 : $", format(round(median(df2025$gross_tax)), big.mark = ","), "\n")
cat("  p-value       :", round(t2$p.value, 6), "\n")
cat("  Decision      :", ifelse(t2$p.value < 0.05,
    "Reject H0 — gross tax significantly increased (p < 0.05)",
    "Fail to reject H0 — composition effect likely; refer to H5"), "\n")


# -- 4. H3 — Do 3+ Bedrooms Command Higher Values? ----------------------------
# H0: μ(value | 3+ beds) ≤ μ(value | < 3 beds)
# H1: μ(value | 3+ beds) > μ(value | < 3 beds)   [one-tailed]
#
# Direction note: t.test() sorts group levels alphabetically, so
# "3+ bedrooms" (group 1) comes before "< 3 bedrooms" (group 2).
# alternative = "less" tests group1 < group2, which is equivalent to
# testing that 3+ bedroom properties have higher values.

cat("\n── H3: Do 3+ bedroom properties command higher assessed values? ──\n")

df2025_beds <- df2025 %>%
  filter(bed_rms >= 1, bed_rms <= 8) %>%
  mutate(bedroom_group = ifelse(bed_rms >= 3, "3+ bedrooms", "< 3 bedrooms"))

t3 <- t.test(total_value ~ bedroom_group, data = df2025_beds, alternative = "less")

cat("  Mean < 3 bedrooms : $",
    format(round(mean(df2025_beds$total_value[df2025_beds$bedroom_group == "< 3 bedrooms"])),
           big.mark = ","), "\n")
cat("  Mean 3+ bedrooms  : $",
    format(round(mean(df2025_beds$total_value[df2025_beds$bedroom_group == "3+ bedrooms"])),
           big.mark = ","), "\n")
cat("  p-value           :", round(t3$p.value, 6), "\n")
cat("  Decision          :", ifelse(t3$p.value < 0.05,
    "Reject H0 — 3+ bedrooms command significantly higher values (p < 0.05)",
    "Fail to reject H0"), "\n")


# -- 5. H4 — Owner-Occupied vs Investor-Owned ---------------------------------
# H0: μ(value | owner-occupied) = μ(value | non-owner-occupied)
# H1: μ differs between groups   [two-tailed]
#
# Non-owner properties are expected to be higher-valued on average as they
# include commercial, investment, and high-density residential assets.

cat("\n── H4: Do owner-occupied properties differ in assessed value? ──\n")

df2025_occ <- df2025 %>% filter(own_occ %in% c("Y", "N"))

t4 <- t.test(total_value ~ own_occ, data = df2025_occ)

cat("  n (non-owner) :", sum(df2025_occ$own_occ == "N"), "\n")
cat("  n (owner)     :", sum(df2025_occ$own_occ == "Y"), "\n")
cat("  Mean non-owner: $",
    format(round(mean(df2025_occ$total_value[df2025_occ$own_occ == "N"])),
           big.mark = ","), "\n")
cat("  Mean owner    : $",
    format(round(mean(df2025_occ$total_value[df2025_occ$own_occ == "Y"])),
           big.mark = ","), "\n")
cat("  p-value       :", round(t4$p.value, 6), "\n")
cat("  Decision      :", ifelse(t4$p.value < 0.05,
    "Reject H0 — significant value difference by occupancy type (p < 0.05)",
    "Fail to reject H0"), "\n")


# -- 6. H5 — Has the Effective Tax Burden Increased? --------------------------
# H0: μ(tax_to_value_ratio, 2023) ≥ μ(tax_to_value_ratio, 2025)
# H1: μ(tax_to_value_ratio, 2023) < μ(tax_to_value_ratio, 2025)  [one-tailed]
#
# This is the most policy-relevant test: if the ratio rose, homeowners
# are paying a larger share of their property value in taxes each year,
# regardless of composition differences between the two samples.

cat("\n── H5: Has the effective tax burden (tax-to-value ratio) increased? ──\n")

t5 <- t.test(df2023$tax_to_value_ratio, df2025$tax_to_value_ratio,
             alternative = "less")

cat("  Median FY2023 :", round(median(df2023$tax_to_value_ratio, na.rm = TRUE), 4), "%\n")
cat("  Median FY2025 :", round(median(df2025$tax_to_value_ratio, na.rm = TRUE), 4), "%\n")
cat("  p-value       :", round(t5$p.value, 6), "\n")
cat("  Decision      :", ifelse(t5$p.value < 0.05,
    "Reject H0 — tax burden significantly increased 2023 to 2025 (p < 0.05)",
    "Fail to reject H0"), "\n")


# -- 7. ANOVA — Does Building Type Affect Property Value? ---------------------
# H0: μ(value) is equal across all building types
# H1: At least one building type has a significantly different mean value
#
# Restricted to types with n ≥ 200 for stable estimates. Tukey HSD
# post-hoc controls the family-wise error rate across pairwise comparisons.

cat("\n── ANOVA: Does building type significantly affect property value? ──\n")

top_types <- df2025 %>%
  count(bldg_type_clean, sort = TRUE) %>%
  filter(n >= 200) %>%
  pull(bldg_type_clean)

cat("Building types included:", length(top_types), "\n")

anova_model   <- aov(total_value ~ bldg_type_clean,
                     data = df2025 %>% filter(bldg_type_clean %in% top_types))
anova_summary <- summary(anova_model)
f_stat        <- anova_summary[[1]]$`F value`[1]
p_val         <- anova_summary[[1]]$`Pr(>F)`[1]

cat("  F-statistic :", round(f_stat, 4), "\n")
cat("  p-value     :", round(p_val, 8), "\n")
cat("  Decision    :", ifelse(p_val < 0.05,
    "Reject H0 — building type significantly affects value (p < 0.05)",
    "Fail to reject H0"), "\n")

cat("\nTukey HSD pairwise comparisons:\n")
print(TukeyHSD(anova_model))


# -- 8. Supplementary: Price per Square Foot ----------------------------------

cat("\n── Supplementary: Price per Square Foot — FY2023 vs FY2025 ──\n")

t_psf <- t.test(df2023$price_per_sqft, df2025$price_per_sqft)

cat("  Mean FY2023 : $", round(mean(df2023$price_per_sqft), 2), "/ sq ft\n")
cat("  Mean FY2025 : $", round(mean(df2025$price_per_sqft), 2), "/ sq ft\n")
cat("  p-value     :", round(t_psf$p.value, 6), "\n")
cat("  Decision    :", ifelse(t_psf$p.value < 0.05,
    "Significant change in price per sq ft (p < 0.05)",
    "No significant change"), "\n")


# -- 9. Results Summary -------------------------------------------------------

cat("\n── Hypothesis Testing Summary ──\n")

results <- data.frame(
  Hypothesis = c(
    "H1: Property values changed",
    "H2: Gross tax increased",
    "H3: 3+ bedrooms command higher values",
    "H4: Owner-occupied differ in value",
    "H5: Tax-to-value ratio increased",
    "ANOVA: Building type affects value"
  ),
  p_value = round(c(t1$p.value, t2$p.value, t3$p.value,
                    t4$p.value, t5$p.value, p_val), 6),
  Decision = c(
    ifelse(t1$p.value < 0.05, "Reject H0", "Fail to Reject H0"),
    ifelse(t2$p.value < 0.05, "Reject H0", "Fail to Reject H0"),
    ifelse(t3$p.value < 0.05, "Reject H0", "Fail to Reject H0"),
    ifelse(t4$p.value < 0.05, "Reject H0", "Fail to Reject H0"),
    ifelse(t5$p.value < 0.05, "Reject H0", "Fail to Reject H0"),
    ifelse(p_val      < 0.05, "Reject H0", "Fail to Reject H0")
  )
)

print(results)
cat("\nHypothesis testing complete.\n")

