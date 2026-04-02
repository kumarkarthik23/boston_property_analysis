# Data Sources

## City of Boston Property Assessment Data

Both datasets are published by the City of Boston through the Analyze Boston open data portal and are made available under the [Open Data Commons Public Domain Dedication and Licence (ODC-PDDL)](https://opendatacommons.org/licenses/pddl/1-0/).

---

### FY2023 Property Assessment

| Field | Detail |
|-------|--------|
| File | `FY2023_property_assessment.csv` |
| Publisher | City of Boston, Assessing Department |
| Source | [Analyze Boston — Property Assessment FY2023](https://data.boston.gov/dataset/property-assessment/resource/4b99718b-d064-471f-9b3a-74759f6f0afc) |
| Records | 178,598 |
| Columns | 64 |
| Coverage | All taxable properties in Boston assessed as of 1 January 2023 |
| Accessed | 2025 |

---

### FY2025 Property Assessment

| Field | Detail |
|-------|--------|
| File | `FY2025_property_assessment.csv` |
| Publisher | City of Boston, Assessing Department |
| Source | [Analyze Boston — Property Assessment FY2025](https://data.boston.gov/dataset/property-assessment/resource/695a8596-5458-442b-a017-7cd72471aade) |
| Records | 183,445 |
| Columns | 66 |
| Coverage | All taxable properties in Boston assessed as of 1 January 2025 |
| Accessed | 2025 |

---

## Schema Notes

The two datasets share a common core schema but differ in several column names and include new fields in FY2025. All differences are handled programmatically in the analysis scripts.

| Column | FY2023 Name | FY2025 Name |
|--------|-------------|-------------|
| ZIP code | `zipcode` | `zip_code` |
| Kitchens | `kitchen` | `kitchens` |
| Fireplaces | `fire_place` | `fireplaces` |
| Heat | `heat_fuel` | `heat_system` |
| Gross tax header | `GROSS_TAX` | ` GROSS_TAX ` (leading/trailing spaces + `$` prefix) |
| SFYI value | Not present | `sfyi_value` |
| Corner unit | Not present | `corner_unit` |

Numeric columns in both datasets are stored as comma-formatted strings (e.g. `"1,150"`) and are converted to numeric type during loading in all scripts.

---

## Known Data Quality Issues

| Issue | Affected Column(s) | Handling |
|-------|--------------------|----------|
| Properties with `total_value = 0` | `total_value` | Excluded from all models and comparative analyses |
| Impossible build years (< 1700 or > assessment year) | `yr_built` | Excluded from all analyses |
| Near-zero `living_area` producing extreme `price_per_sqft` | `living_area` | Filter `living_area > 0` applied; `price_per_sqft` capped at $5,000/sqft in hypothesis testing |
| `heat_fuel` (FY2023) is 99.5% missing | `heat_fuel` | Dropped — exceeds 50% missingness threshold in all models |
| `gross_tax` header has leading/trailing spaces in FY2025 CSV | `gross_tax` | Stripped during column name standardisation in all scripts |
| `unit_num` contains apartment identifiers (e.g. "1A", "2B") that cannot be cast to numeric | `unit_num` | Coercion suppressed via `suppressWarnings()`; column excluded from all models |

---

## Citation

City of Boston, Assessing Department. *Property Assessment FY2023*. Analyze Boston Open Data Portal.
https://data.boston.gov/dataset/property-assessment

City of Boston, Assessing Department. *Property Assessment FY2025*. Analyze Boston Open Data Portal.
https://data.boston.gov/dataset/property-assessment