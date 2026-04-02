# ============================================================================
# Boston Property Assessment — Large-Scale Analysis with PySpark
# ============================================================================
#
# Author  : Kumar Karthik Ankasandra Naveen
# Data    : City of Boston FY2025 Property Assessment
#           https://data.boston.gov/dataset/property-assessment
#
# Purpose : Distributed analysis of the FY2025 Boston property assessment
#           dataset using Apache Spark 4.1.1. While the 183,445-record dataset
#           fits in memory, PySpark demonstrates production-scale data
#           engineering patterns that generalise to datasets orders of
#           magnitude larger without code changes.
#
# Confirmed results (FY2025):
#   Records          : 183,445 | Columns : 66 | Duplicate rows : 0
#   Avg total value  : $1,542,565
#   Mean price/sqft  : $586.37 (stddev $375.49)
#   Unique cities    : 19 | Unique ZIP codes : 37
#   Top city by value: Boston ($3.25M avg, n = 48,036)
#   Highest tax burden: Roxbury (1.307% avg tax-to-value ratio)
#
# Key fixes for Spark 4.x compatibility:
#   isnan() guard  — isnan() raises CAST_INVALID_INPUT on string columns in
#                    Spark 4.x. Applied only to numeric columns (identified
#                    from schema) via a selective when() expression.
#   Empty string cast — After regexp_replace(), empty strings cannot be cast
#                    to DoubleType in Spark 4.x. A when(== "", None) guard
#                    converts them to NULL before casting.
#   spark_when alias — pyspark.sql.functions.when is aliased as spark_when
#                    to avoid shadowing Python's built-in when-like patterns
#                    and to make intent explicit in the numeric conversion loop.
#
# Analyses:
#   1.  Spark session initialisation
#   2.  Data loading and schema inspection
#   3.  Column name standardisation
#   4.  Numeric type conversion
#   5.  Data quality assessment (missing values, duplicates)
#   6.  Descriptive statistics
#   7.  Feature engineering (price per sqft, age, tax ratio)
#   8.  Property value by city (top 15)
#   9.  Property value by building type
#   10. Price per square foot summary
#   11. Urban development trends (year built distribution)
#   12. Bedroom value premium
#   13. Taxation fairness by city
#   14. Dataset overview summary
#   15. Spark session teardown
#
# Visualisation note: aggregated results are converted to Pandas for
#   matplotlib. This is appropriate because aggregated DataFrames contain
#   at most a few hundred rows — negligible driver-side memory footprint.
#
# File path: script is run from python_analysis/; data lives in data/
# ============================================================================

import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql       import SparkSession
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import (
    col, avg, count, stddev, min, max, sum,
    countDistinct, isnan,
    regexp_replace,
    when    as spark_when,   # Aliased: used in numeric conversion loop
    round   as spark_round,  # Aliased: avoids shadowing Python's built-in round()
)


# -- 1. Spark Session ---------------------------------------------------------
# shuffle.partitions is reduced from the default 200 to 8.
# On a single-node local machine more partitions add scheduling overhead
# without any parallelism benefit — 8 matches the available CPU parallelism.

spark = (
    SparkSession.builder
    .appName("BostonPropertyAssessment_FY2025")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print(f"Spark {spark.version} | Local parallelism: {spark.sparkContext.defaultParallelism}")


# -- 2. Data Loading ----------------------------------------------------------
# inferSchema=True auto-detects column types from the first pass of the file.
# GROSS_TAX and several value columns will be inferred as StringType due to
# "$" prefix and comma formatting — these are corrected in Section 4.

FILE_PATH = "../data/FY2025_property_assessment.csv"

df = spark.read.csv(FILE_PATH, header=True, inferSchema=True)

df.printSchema()
print(f"Records: {df.count():,} | Columns: {len(df.columns)}")


# -- 3. Column Name Standardisation ------------------------------------------
# Strip whitespace and lowercase all column headers. The FY2025 CSV stores
# gross_tax as " GROSS_TAX " (with leading and trailing spaces) — this
# normalisation ensures the column can be referenced by name reliably.

new_names = [c.strip().lower().replace(" ", "_") for c in df.columns]
df        = df.toDF(*new_names)

# Catch residual gross_tax naming variants after renaming
tax_candidates = [c for c in df.columns if "gross_tax" in c]
if tax_candidates and tax_candidates[0] != "gross_tax":
    df = df.withColumnRenamed(tax_candidates[0], "gross_tax")

print(f"Columns standardised ({len(df.columns)} total)")


# -- 4. Numeric Type Conversion -----------------------------------------------
# regexp_replace() strips commas and currency symbols; withColumn() replaces
# the original column in place. All operations execute inside Spark —
# no data is pulled to the driver here.
#
# Empty string guard:
#   After stripping formatting characters, fields that were NULL or contained
#   only non-numeric text produce an empty string "". Spark 4.x raises a
#   hard CAST_INVALID_INPUT error when casting "" to DoubleType.
#   spark_when(cleaned == "", None) converts empty strings to NULL so Spark
#   treats them as missing values rather than malformed input.

STRING_NUMERIC_COLS = [
    "land_sf", "land_value", "bldg_value",
    "total_value", "gross_area", "living_area",
]

for col_name in STRING_NUMERIC_COLS:
    if col_name in df.columns:
        cleaned = regexp_replace(col(col_name).cast("string"), ",", "")
        df = df.withColumn(
            col_name,
            spark_when(cleaned == "", None).otherwise(cleaned.cast(DoubleType()))
        )

# gross_tax: strip "$", commas, and any other non-numeric characters
cleaned_tax = regexp_replace(col("gross_tax").cast("string"), r"[^0-9.]", "")
df = df.withColumn(
    "gross_tax",
    spark_when(cleaned_tax == "", None).otherwise(cleaned_tax.cast(DoubleType()))
)

print("Numeric conversion complete | sample values:")
df.select("total_value", "land_sf", "gross_tax").show(3)


# -- 5. Data Quality Assessment -----------------------------------------------
# isnull() catches SQL NULLs; isnan() catches floating-point NaN values.
# isnan() only accepts numeric columns — calling it on a string column raises
# CAST_INVALID_INPUT in Spark 4.x. The schema is inspected to build a set of
# numeric column names, and isnan() is applied only within that set.

print("\n-- Missing Values per Column (non-zero only) --")

NUMERIC_TYPES    = {"double", "float", "int", "bigint", "long", "decimal"}
numeric_cols_set = {
    f.name for f in df.schema.fields
    if any(t in str(f.dataType).lower() for t in NUMERIC_TYPES)
}

missing_expr = [
    sum(
        spark_when(
            col(c).isNull() | (isnan(col(c)) if c in numeric_cols_set else col(c).isNull()),
            1
        ).otherwise(0)
    ).alias(c)
    for c in df.columns
]

missing_pd = (
    df.select(missing_expr)
    .toPandas().T
    .rename(columns={0: "missing_count"})
)
missing_pd = missing_pd[missing_pd["missing_count"] > 0].sort_values(
    "missing_count", ascending=False
)
print(missing_pd.to_string())

total_rows    = df.count()
distinct_rows = df.dropDuplicates().count()
print(f"\nTotal rows     : {total_rows:,}")
print(f"Distinct rows  : {distinct_rows:,}")
print(f"Duplicate rows : {total_rows - distinct_rows:,}")


# -- 6. Descriptive Statistics ------------------------------------------------
# describe() computes count, mean, stddev, min, and max for selected columns.
# Results are computed lazily and triggered by .show() — equivalent to
# R's summary() or pandas .describe().

print("\n-- Descriptive Statistics (Key Numeric Columns) --")
df.select(
    "total_value", "land_value", "bldg_value",
    "gross_tax", "land_sf", "living_area",
    "bed_rms", "full_bth", "yr_built"
).describe().show()


# -- 7. Feature Engineering ---------------------------------------------------
# withColumn() adds each derived feature in a single chained transformation.
# spark_round() is used instead of Python's built-in round() to keep all
# computations inside the Spark execution plan.

CURRENT_YEAR = 2025

df = (
    df
    .withColumn(
        "price_per_sqft",
        # Per-sqft rate normalises value by interior size for cross-area comparison
        spark_round(col("total_value") / col("living_area"), 2)
    )
    .withColumn(
        "age_of_property",
        # Proxy for depreciation — older properties generally worth less per sqft
        (CURRENT_YEAR - col("yr_built")).cast(IntegerType())
    )
    .withColumn(
        "tax_to_value_ratio",
        # Effective tax burden per dollar of assessed value
        # Higher values indicate heavier proportional taxation
        spark_round(col("gross_tax") / col("total_value") * 100, 4)
    )
)

print("\nEngineered features added | sample:")
df.select("total_value", "price_per_sqft", "age_of_property",
          "tax_to_value_ratio").show(5)


# -- 8. Property Value by City ------------------------------------------------
# Restricted to cities with ≥ 100 properties to suppress small-sample noise.
# Boston proper dominates at $3.25M avg value, nearly 3x the next city.

print("\n-- Average Property Value by City (Top 15) --")
(
    df
    .filter(col("total_value").isNotNull())
    .groupBy("city")
    .agg(
        avg("total_value").alias("avg_value"),
        count("*").alias("n"),
        avg("gross_tax").alias("avg_tax"),
        avg("price_per_sqft").alias("avg_psf"),
    )
    .filter(col("n") >= 100)
    .orderBy(col("avg_value").desc())
    .show(15, truncate=False)
)


# -- 9. Property Value by Building Type ---------------------------------------
# Ordered by property count to show the most common types first.
# High-rise buildings command the highest average values; vacant lots the lowest.

print("\n-- Average Property Value by Building Type --")
(
    df
    .filter(col("total_value").isNotNull() & col("bldg_type").isNotNull())
    .groupBy("bldg_type")
    .agg(
        avg("total_value").alias("avg_value"),
        count("*").alias("n"),
    )
    .orderBy(col("n").desc())
    .show(15, truncate=False)
)


# -- 10. Price Per Square Foot Summary ----------------------------------------
# Capped at $10,000/sqft to exclude data entry errors caused by properties
# with near-zero living_area (confirmed mean: $586/sqft, stddev: $375).

print("\n-- Price Per Square Foot — Summary Statistics --")
(
    df
    .filter(
        col("price_per_sqft").isNotNull() &
        (col("price_per_sqft") > 0) &
        (col("price_per_sqft") < 10000)
    )
    .agg(
        avg("price_per_sqft").alias("mean_psf"),
        stddev("price_per_sqft").alias("stddev_psf"),
        min("price_per_sqft").alias("min_psf"),
        max("price_per_sqft").alias("max_psf"),
    )
    .show()
)


# -- 11. Urban Development Trends ---------------------------------------------
# Aggregated year-built counts converted to Pandas for matplotlib.
# The resulting DataFrame has ~300 rows — negligible driver-side footprint.
# Peak construction circa 1880-1920 reflects Boston's historical period of
# rapid urbanisation and the triple-decker building boom.

print("\n-- Property Count by Year Built --")

year_dist_pd = (
    df
    .filter(
        col("yr_built").isNotNull() &
        (col("yr_built") > 1700) &
        (col("yr_built") <= 2025)
    )
    .groupBy("yr_built")
    .count()
    .orderBy("yr_built")
    .toPandas()
)

plt.figure(figsize=(14, 5))
plt.bar(year_dist_pd["yr_built"], year_dist_pd["count"],
        color="#4A90D9", alpha=0.85, width=1)
plt.title("Properties by Year Built — FY2025\n"
          "Peak circa 1880–1920 reflects Boston's historical development period",
          fontsize=13, pad=12)
plt.xlabel("Year Built")
plt.ylabel("Property Count")
plt.tight_layout()
plt.show()


# -- 12. Bedroom Value Premium ------------------------------------------------
# The positive slope of this line is the incremental value of each additional
# bedroom — consistent with H3 in hypothesis_tests.R which confirmed that
# 3+ bedroom properties command significantly higher values (p ≈ 0).

print("\n-- Average Property Value by Number of Bedrooms --")

bedroom_pd = (
    df
    .filter(
        col("bed_rms").isNotNull() &
        (col("bed_rms") >= 1) &
        (col("bed_rms") <= 8)   # Cap at 8 to exclude anomalous multi-unit records
    )
    .groupBy("bed_rms")
    .agg(
        avg("total_value").alias("avg_value"),
        count("*").alias("n"),
    )
    .orderBy("bed_rms")
    .toPandas()
)

print(bedroom_pd.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(bedroom_pd["bed_rms"], bedroom_pd["avg_value"] / 1e6,
         marker="o", color="#4A90D9", linewidth=2)
plt.title("Average Property Value by Number of Bedrooms — FY2025",
          fontsize=13, pad=12)
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Value ($M)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -- 13. Taxation Fairness by City -------------------------------------------
# A higher tax-to-value ratio means homeowners pay more per dollar of assessed
# value — a direct measure of taxation equity across neighbourhoods.
# Capped at 10% to exclude extreme outliers. Results consistent with H5 in
# hypothesis_tests.R: citywide ratio rose from 1.088% (FY2023) to 1.158% (FY2025).

print("\n-- Effective Tax Burden by City (Tax-to-Value Ratio, Top 15) --")
(
    df
    .filter(
        col("tax_to_value_ratio").isNotNull() &
        (col("tax_to_value_ratio") > 0) &
        (col("tax_to_value_ratio") < 10)
    )
    .groupBy("city")
    .agg(
        avg("tax_to_value_ratio").alias("avg_tax_ratio"),
        count("*").alias("n"),
    )
    .filter(col("n") >= 100)
    .orderBy(col("avg_tax_ratio").desc())
    .show(15, truncate=False)
)


# -- 14. Dataset Overview Summary ---------------------------------------------

print("\n-- Dataset Overview — FY2025 --")
(
    df.select(
        avg("total_value").alias("avg_total_value"),
        avg("gross_area").alias("avg_gross_area"),
        avg("land_value").alias("avg_land_value"),
        avg("bldg_value").alias("avg_bldg_value"),
        countDistinct("lu_desc").alias("unique_property_types"),
        countDistinct("city").alias("unique_cities"),
        countDistinct("zip_code").alias("unique_zip_codes"),
    )
    .show()
)


# -- 15. Spark Session Teardown -----------------------------------------------
# Stopping the session releases cluster resources and closes the Spark UI
# on port 4040. Always called explicitly — relying on garbage collection
# can leave orphaned processes.

spark.stop()
print("\nSpark session stopped. PySpark analysis complete.")