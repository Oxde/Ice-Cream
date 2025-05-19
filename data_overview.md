# Marketing Mix Modelling Data Overview
## Dataset Summary
| Dataset | Rows | Columns | Time Range | Missing Values |
|---------|------|---------|------------|----------------|
| email | 104 | 2 | 2022-01-03 to 2023-12-25 | 0 |
| sales | 260 | 2 | 2020-01-06 to 2024-12-23 | 0 |
| tv_promo | 260 | 3 | 2020-01-06 to 2024-12-23 | 0 |
| tv_branding | 260 | 7 | 2020-01-06 to 2024-12-23 | 1039 |
| radio_local | 260 | 3 | 2020-01-06 to 2024-12-23 | 0 |
| social | 260 | 3 | 2020-01-06 to 2024-12-23 | 0 |
| radio_national | 260 | 3 | 2020-01-06 to 2024-12-23 | 0 |
| promo_wide | 3 | 82 | Not found | 82 |
| search | 260 | 3 | 2020-01-06 to 2024-12-08 | 0 |
| ooh | 260 | 2 | 2020-01-06 to 2024-12-23 | 0 |

## Detailed Dataset Information

### email
- **File:** email.xlsx
- **Rows:** 104
- **Columns:** date, email_campaigns
- **Time Range:** 2022-01-03 to 2023-12-25 (Column: date)

**Sample Data:**

| date | email_campaigns |
| --- | --- |
| 2022-01-03 00:00:00 | 1 |
| 2022-01-10 00:00:00 | 0 |
| 2022-01-17 00:00:00 | 0 |
| 2022-01-24 00:00:00 | 2 |
| 2022-01-31 00:00:00 | 1 |

### sales
- **File:** sales.xlsx
- **Rows:** 260
- **Columns:** date, sales
- **Time Range:** 2020-01-06 to 2024-12-23 (Column: date)

**Sample Data:**

| date | sales |
| --- | --- |
| 2020-01-06 00:00:00 | 105859 |
| 2020-01-13 00:00:00 | 113066 |
| 2020-01-20 00:00:00 | 116292 |
| 2020-01-27 00:00:00 | 119485 |
| 2020-02-03 00:00:00 | 125162 |

### tv_promo
- **File:** tv_promo.xlsx
- **Rows:** 260
- **Columns:** datum, tv_promo_grps, tv_promo_cost
- **Time Range:** 2020-01-06 to 2024-12-23 (Column: datum)

**Sample Data:**

| datum | tv_promo_grps | tv_promo_cost |
| --- | --- | --- |
| 2020-01-06 00:00:00 | 46.61 | 1864.48 |
| 2020-01-13 00:00:00 | 90.87 | 3634.69 |
| 2020-01-20 00:00:00 | 66.75 | 2669.90 |
| 2020-01-27 00:00:00 | 91.41 | 3656.48 |
| 2020-02-03 00:00:00 | 64.73 | 2589.39 |

### tv_branding
- **File:** tv_branding.xlsx
- **Rows:** 260
- **Columns:** datum, tv_branding_grps, tv_branding_cost, Unnamed: 3, Unnamed: 4, Unnamed: 5, Unnamed: 6
- **Time Range:** 2020-01-06 to 2024-12-23 (Column: datum)
- **Missing Values:**
  - Unnamed: 3: 260 missing values
  - Unnamed: 4: 260 missing values
  - Unnamed: 5: 260 missing values
  - Unnamed: 6: 259 missing values

**Sample Data:**

| datum | tv_branding_grps | tv_branding_cost | Unnamed: 3 | Unnamed: 4 | Unnamed: 5 | Unnamed: 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 2020-01-06 00:00:00 | 92.19 | 4148.64 | NA | NA | NA | NA |
| 2020-01-13 00:00:00 | 118.21 | 5319.64 | NA | NA | NA | Belangrijk: tv grp's zijn t/m december 2021 inschattingen en geen actuals |
| 2020-01-20 00:00:00 | 60.00 | 2700.00 | NA | NA | NA | NA |
| 2020-01-27 00:00:00 | 89.27 | 4017.08 | NA | NA | NA | NA |
| 2020-02-03 00:00:00 | 112.42 | 5059.03 | NA | NA | NA | NA |

### radio_local
- **File:** radio_local.xlsx
- **Rows:** 260
- **Columns:** dag, radio_local_grps, radio_local_cost
- **Time Range:** 2020-01-06 to 2024-12-23 (Column: dag)

**Sample Data:**

| dag | radio_local_grps | radio_local_cost |
| --- | --- | --- |
| 2020-01-06 00:00:00 | 94.44 | 1888.85 |
| 2020-01-13 00:00:00 | 56.62 | 1132.33 |
| 2020-01-20 00:00:00 | 99.50 | 1990.10 |
| 2020-01-27 00:00:00 | 105.26 | 2105.29 |
| 2020-02-03 00:00:00 | 103.99 | 2079.90 |

### social
- **File:** social.xlsx
- **Rows:** 260
- **Columns:** date, impressions, costs
- **Time Range:** 2020-01-06 to 2024-12-23 (Column: date)

**Sample Data:**

| date | impressions | costs |
| --- | --- | --- |
| 2020-01-06 00:00:00 | 99466 | 596.79 |
| 2020-01-13 00:00:00 | 81320 | 487.92 |
| 2020-01-20 00:00:00 | 50000 | 300.00 |
| 2020-01-27 00:00:00 | 122101 | 732.61 |
| 2020-02-03 00:00:00 | 118421 | 710.53 |

### radio_national
- **File:** radio_national.xlsx
- **Rows:** 260
- **Columns:** dag, radio_national_grps, radio_national_cost
- **Time Range:** 2020-01-06 to 2024-12-23 (Column: dag)

**Sample Data:**

| dag | radio_national_grps | radio_national_cost |
| --- | --- | --- |
| 2020-01-06 00:00:00 | 93.68 | 2341.96 |
| 2020-01-13 00:00:00 | 50.82 | 1270.56 |
| 2020-01-20 00:00:00 | 81.57 | 2039.34 |
| 2020-01-27 00:00:00 | 59.23 | 1480.75 |
| 2020-02-03 00:00:00 | 56.55 | 1413.69 |

### promo_wide
- **File:** promo_wide.xlsx
- **Rows:** 3
- **Columns:** Week, 1, 4, 9, 13, 9.1, 13.1, 2, 3, 5, 8, 9.2, 13.2, 5.1, 7, 9.3, 2.1, 8.1, 9.4, 5.2, 10, 11, 12, 1.1, 3.1, 5.3, 6, 7.1, 10.1, 12.1, 2.2, 7.2, 9.5, 10.2, 12.2, 2.3, 8.2, 6.1, 7.3, 9.6, 10.3, 13.3, 1.2, 3.2, 5.4, 6.2, 8.3, 11.1, 4.1, 8.4, 9.7, 10.4, 11.2, 12.3, 3.3, 4.2, 6.3, 2.4, 3.4, 4.3, 7.4, 12.4, 6.4, 10.5, 12.5, 1.3, 2.5, 7.5, 11.3, 2.6, 8.5, 9.8, 11.4, 13.4, 1.4, 3.5, 8.6, 9.9, 11.5, 12.6, 3.6, 5.5
- **Missing Values:**
  - Week: 1 missing values
  - 1: 1 missing values
  - 4: 1 missing values
  - 9: 1 missing values
  - 13: 1 missing values
  - 9.1: 1 missing values
  - 13.1: 1 missing values
  - 2: 1 missing values
  - 3: 1 missing values
  - 5: 1 missing values
  - 8: 1 missing values
  - 9.2: 1 missing values
  - 13.2: 1 missing values
  - 5.1: 1 missing values
  - 7: 1 missing values
  - 9.3: 1 missing values
  - 2.1: 1 missing values
  - 8.1: 1 missing values
  - 9.4: 1 missing values
  - 5.2: 1 missing values
  - 10: 1 missing values
  - 11: 1 missing values
  - 12: 1 missing values
  - 1.1: 1 missing values
  - 3.1: 1 missing values
  - 5.3: 1 missing values
  - 6: 1 missing values
  - 7.1: 1 missing values
  - 10.1: 1 missing values
  - 12.1: 1 missing values
  - 2.2: 1 missing values
  - 7.2: 1 missing values
  - 9.5: 1 missing values
  - 10.2: 1 missing values
  - 12.2: 1 missing values
  - 2.3: 1 missing values
  - 8.2: 1 missing values
  - 6.1: 1 missing values
  - 7.3: 1 missing values
  - 9.6: 1 missing values
  - 10.3: 1 missing values
  - 13.3: 1 missing values
  - 1.2: 1 missing values
  - 3.2: 1 missing values
  - 5.4: 1 missing values
  - 6.2: 1 missing values
  - 8.3: 1 missing values
  - 11.1: 1 missing values
  - 4.1: 1 missing values
  - 8.4: 1 missing values
  - 9.7: 1 missing values
  - 10.4: 1 missing values
  - 11.2: 1 missing values
  - 12.3: 1 missing values
  - 3.3: 1 missing values
  - 4.2: 1 missing values
  - 6.3: 1 missing values
  - 2.4: 1 missing values
  - 3.4: 1 missing values
  - 4.3: 1 missing values
  - 7.4: 1 missing values
  - 12.4: 1 missing values
  - 6.4: 1 missing values
  - 10.5: 1 missing values
  - 12.5: 1 missing values
  - 1.3: 1 missing values
  - 2.5: 1 missing values
  - 7.5: 1 missing values
  - 11.3: 1 missing values
  - 2.6: 1 missing values
  - 8.5: 1 missing values
  - 9.8: 1 missing values
  - 11.4: 1 missing values
  - 13.4: 1 missing values
  - 1.4: 1 missing values
  - 3.5: 1 missing values
  - 8.6: 1 missing values
  - 9.9: 1 missing values
  - 11.5: 1 missing values
  - 12.6: 1 missing values
  - 3.6: 1 missing values
  - 5.5: 1 missing values

**Sample Data:**

| Week | 1 | 4 | 9 | 13 | 9.1 | 13.1 | 2 | 3 | 5 | 8 | 9.2 | 13.2 | 5.1 | 7 | 9.3 | 2.1 | 8.1 | 9.4 | 5.2 | 10 | 11 | 12 | 1.1 | 3.1 | 5.3 | 6 | 7.1 | 10.1 | 12.1 | 2.2 | 7.2 | 9.5 | 10.2 | 12.2 | 2.3 | 8.2 | 6.1 | 7.3 | 9.6 | 10.3 | 13.3 | 1.2 | 3.2 | 5.4 | 6.2 | 8.3 | 11.1 | 4.1 | 8.4 | 9.7 | 10.4 | 11.2 | 12.3 | 3.3 | 4.2 | 6.3 | 2.4 | 3.4 | 4.3 | 7.4 | 12.4 | 6.4 | 10.5 | 12.5 | 1.3 | 2.5 | 7.5 | 11.3 | 2.6 | 8.5 | 9.8 | 11.4 | 13.4 | 1.4 | 3.5 | 8.6 | 9.9 | 11.5 | 12.6 | 3.6 | 5.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NA | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2020 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2021 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2022 | 2023 | 2023 | 2023 | 2023 | 2023 | 2023 | 2023 | 2023 | 2023 | 2023 | 2023 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 | 2024 |
| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| NA | Buy One Get One | Limited Time Offer | Limited Time Offer | Price Discount | Price Discount | Limited Time Offer | Buy One Get One | Buy One Get One | Price Discount | Price Discount | Limited Time Offer | Limited Time Offer | Limited Time Offer | Limited Time Offer | Price Discount | Price Discount | Price Discount | Limited Time Offer | Buy One Get One | Price Discount | Buy One Get One | Price Discount | Price Discount | Price Discount | Price Discount | Price Discount | Price Discount | Price Discount | Price Discount | Buy One Get One | Price Discount | Price Discount | Price Discount | Price Discount | Price Discount | Limited Time Offer | Buy One Get One | Buy One Get One | Price Discount | Price Discount | Price Discount | Buy One Get One | Price Discount | Limited Time Offer | Price Discount | Limited Time Offer | Buy One Get One | Price Discount | Limited Time Offer | Buy One Get One | Buy One Get One | Buy One Get One | Buy One Get One | Limited Time Offer | Limited Time Offer | Price Discount | Limited Time Offer | Limited Time Offer | Buy One Get One | Price Discount | Limited Time Offer | Buy One Get One | Limited Time Offer | Price Discount | Limited Time Offer | Price Discount | Limited Time Offer | Price Discount | Buy One Get One | Buy One Get One | Buy One Get One | Price Discount | Price Discount | Price Discount | Price Discount | Buy One Get One | Price Discount | Limited Time Offer | Price Discount | Price Discount | Buy One Get One |

### search
- **File:** search.xlsx
- **Rows:** 260
- **Columns:** date, impressions, cost
- **Time Range:** 2020-01-06 to 2024-12-08 (Column: date)

**Sample Data:**

| date | impressions | cost |
| --- | --- | --- |
| 2020-06-01 00:00:00 | 129957 | 519.83 |
| NA | 179761 | 719.05 |
| NA | 144751 | 579.00 |
| NA | 127328 | 509.31 |
| 2020-03-02 00:00:00 | 166095 | 664.38 |

### ooh
- **File:** ooh.xlsx
- **Rows:** 260
- **Columns:** date, ooh_spend
- **Time Range:** 2020-01-06 to 2024-12-23 (Column: date)

**Sample Data:**

| date | ooh_spend |
| --- | --- |
| 2020-01-06 00:00:00 | 780.31 |
| 2020-01-13 00:00:00 | 811.53 |
| 2020-01-20 00:00:00 | 766.27 |
| 2020-01-27 00:00:00 | 702.50 |
| 2020-02-03 00:00:00 | 825.30 |
