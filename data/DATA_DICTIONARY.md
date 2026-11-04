# Data dictionary

Blank numeric cells mean unobserved values; never replace them with zero before
constructing masks. Hb = haemoglobin concentration, not HbA1c.

## daily_records.csv

| Column | Type / unit | Meaning |
|---|---|---|
| participant_id | text | Fictional ID, stable across dates and alternative scenarios |
| date | ISO date | Calendar observation date |
| hb | g/dL | Synthetic observed Hb, usually missing |
| weight | kg | Synthetic observed weight, usually missing |
| steps | steps/day | Synthetic daily total, or missing |
| age | years | Fixed synthetic baseline age, 18-75 |
| gender | category | Synthetic female/male label; simplified generator field |
| split | category | train, validation, calibration or test; exclude from inputs |
| is_synthetic | integer | Always 1; exclude from inputs |

## forecast_examples.csv

Each row is one participant/anchor/horizon example. An anchor may generate two rows.

| Column | Unit / values | Role |
|---|---|---|
| participant_id | text | Grouping identifier only |
| anchor_date | ISO date | Forecast origin, not a future input |
| hb0 | g/dL | Latest measured Hb at origin; input and persistence reference |
| weight0 | kg | Latest measured weight on/before origin; input/reference |
| hb_age_days | days | Always 0 in these examples; baseline input |
| weight_age_days | days | Age of latest weight measurement; input |
| age, gender | years, category | Baseline inputs |
| steps_day_-4 through steps_day_0 | steps/day | Five-day input window, oldest to newest, inclusive of origin |
| steps_missing_day_-4 through steps_missing_day_0 | 0/1 | 1 means corresponding input day missing |
| steps0 | steps/day | Observed-day mean of five input days; persistence reference |
| valid_step_days | count | Number of observed input days, 3-5; optional baseline input |
| horizon_days | 30 or 90 | Forecast horizon; filter for separate-horizon models |
| target_date | ISO date | Origin + horizon; metadata, exclude from X |
| y_hb | g/dL | Observed future Hb target; never an input |
| y_weight | kg | Observed future weight target; never an input |
| y_steps | steps/day | Mean steps over seven days ending at target date; never an input |
| mask_hb, mask_weight, mask_steps | 0/1 | Target observed flags for loss only; exclude from X |
| hb_target_observed_date, weight_target_observed_date | ISO date | Actual future measurement dates; exclude from X |
| hb_target_offset_days, weight_target_offset_days | -7 to +7 days | Actual measurement date minus nominal target date; exclude from X |
| target_step_valid_days | 0-7 | Future observation coverage; exclude from X |
| split, is_synthetic | category, 1 | Provenance only; exclude from X |

## Residuals

Derive residual targets only after splitting: Hb residual = y_hb - hb0;
weight residual = y_weight - weight0; step residual = y_steps - steps0.
Estimate standardisation statistics from observed training targets only. Deriving
residuals does not make missing targets observed. Use one common input feature set
for Hb-only and multitask comparisons. Participant IDs are repeated across alternative
scenarios by design: each scenario is a separate experiment, not additional people.
