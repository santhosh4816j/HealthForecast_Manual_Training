# Synthetic Health Forecasting Dataset v1.0

This is reference documentation for the earlier four-scenario dataset package.
The application package includes only its both_related/forecast_examples.csv,
renamed sample_forecast_examples.csv. The complete generator is included under tools/.

ALL RECORDS ARE SYNTHETIC. No real people, patient records or clinical sources were used.
This dataset supports software development and controlled simulation experiments for
Adaptive Multivariate Health Forecasting / HS-GAW. It cannot validate clinical accuracy,
establish a biological link between steps and Hb, or support medical decisions.

## Start here

Use `both_related/forecast_examples.csv` for your first training experiment.
Use `both_related/daily_records.csv` if your application expects participant_id, date,
hb, weight, steps, age and gender. Daily records are sparse by design: blank Hb on
most days means no measurement, not zero. Keep the participant splits provided.

The package contains 4 alternative simulation worlds, each with the same 400 fictional
participant identifiers and 241 daily records per participant (2024-01-01 to 2024-08-28).
Do not combine worlds into a larger cohort: they reuse exogenous random draws and people.

## Files

- Each scenario: `daily_records.csv` and `forecast_examples.csv`.
- `participants_and_splits.csv`: fixed 240 train / 60 validation / 40 calibration /
  60 test participants, shared across scenarios.
- `generate_synthetic_dataset.py`: complete reproducible generator.
- `requirements.txt`: NumPy and pandas dependencies.
- `DATA_DICTIONARY.md`: columns, units, feature exclusions and target construction.
- `validation_summary.json`: actual counts, missingness and structural check results.
- `checksums.json`: SHA-256 digests of all supplied CSVs.

## Four scenarios

1. `independent_auxiliaries`: Hb has an independent stochastic trajectory. Weight and
   step supervision has no deliberately planted Hb relationship. Shared baseline inputs
   and chance finite-sample associations can still affect results.
2. `weight_related`: Hb and weight share a slowly changing latent factor; steps do not
   deliberately share an Hb factor.
3. `both_related`: Hb shares one latent factor with weight and another with steps.
4. `mixed_timescales`: a larger contribution from a faster activity-related factor and
   a smaller slow weight-related factor allow investigation of horizon-dependent effects.

These names describe the generator's assumptions. They DO NOT establish that multitask
learning helps, harms, or wins at either horizon. A shared latent relationship need not
make auxiliary supervision useful beyond the same input features. Determine outcomes
experimentally. No model was trained to produce this dataset; no accuracy is claimed.

## Data-generating assumptions (arbitrary, not clinical estimates)

Each person has independent stationary Gaussian AR(1) latent trajectories M (rho .995),
A (rho .94) and U (rho .98), with unit stationary variance. These latent variables are
not exported as model inputs. Person-specific Hb, weight and step baselines are uniform
draws. Weight = baseline + 2 M + Gaussian noise (SD .15 kg). Steps = baseline + 1400 A
+ Gaussian noise (SD 650), rounded and clipped to [0,22000]. Hb = baseline + signal +
Gaussian noise (SD .13 g/dL), clipped to [7.5,18.5]. The signal is:

- Independent: .55 U
- Weight-related: .5 M + .3 U
- Both-related: .4 M + .3 A + .3 U
- Mixed timescales: .3 M + .5 A + .3 U

Hb is observed on roughly monthly dates with +/-3-day scheduling jitter and 15%
scheduled-measurement missingness. Weight is observed about every 14 days with +/-2-day
jitter and 10% scheduled missingness. Day zero has both measurements. Daily steps have
10% independent missingness. Missingness is intentionally simple and does not model
informative clinical follow-up, device nonwear or treatment changes.

## Targets and safe training

Forecast anchors are observed Hb dates with five calendar days of step history and at
least three observed step days. The horizon is 30 or 90 days. Future Hb and weight are
the nearest actually observed measurement within +/-7 days of the horizon date (ties
use earlier). Future steps are the observed-day mean across the seven calendar days
ending on that date, requiring at least five observed days. Targets without measurements
remain blank. These are approximate-horizon Hb targets: report the date tolerance.

Use only documented baseline features as X. Never include y_*, mask_*, future dates,
future offsets, target_step_valid_days, split, participant_id or is_synthetic as model
inputs. horizon_days can be an input to a joint horizon model; your report's separate
30- and 90-day models should filter to a single horizon before training.

Fit imputation, input scaling and residual scaling inside each training fold. Keep
missingness flags and mask missing targets in the loss. For Hb-only comparisons, use
identical baseline inputs, architecture, seeds and tuning budgets. Hb-only refers to
the training objective, not removal of weight or step inputs. Some rows lack future
Hb but retain auxiliary targets: document whether you use those rows in training.

Use train participants for fitting and participant-grouped cross-validation; use the
validation participants for a prespecified final development check or selection.
Do not repeatedly tune on calibration or test people. Use calibration people only for
interval calibration. Evaluate Hb MAE per person first, then average across people;
report participant bootstrap intervals and correct comparisons across horizons.
Reusing test results to edit the generator or tune HS-GAW invalidates an untouched test
claim. Include several independent generator seeds and training seeds for a stronger
simulation study; the single supplied seed is a starter benchmark, not complete research.

## Reproduce

Install requirements, then run:

    python generate_synthetic_dataset.py --output regenerated --seed 20261104 --people 400

The generator recreates CSVs, checksums and validation statistics. Documentation is
provided separately in this package. All arithmetic parameters are visible in source.

## Research limitations

No clinical calibration or validation; only adult binary synthetic sex categories;
no treatments, illness episodes, menstrual effects or other clinical mechanisms;
short follow-up; a single sparse measurement schedule; Hb ages are always zero in
the prepared examples because anchors occur on observed Hb dates; weight ages vary.
Do not describe this as a representative patient population or use it to claim that
five days of steps are sufficient for real-world Hb forecasting. Missingness is random
and can underestimate deployment difficulty. Simulation conclusions apply only to the
specified assumptions. Real linked longitudinal records are required for health claims.

Suggested methods wording: "We evaluated the pipeline on fully synthetic longitudinal
records generated under independent and shared-latent-factor assumptions. Reported
performance describes simulation behaviour and does not establish clinical validity."
