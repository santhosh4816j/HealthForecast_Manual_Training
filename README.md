# Adaptive Multivariate Health Forecasting

Three-module localhost application with manual CSV upload and HS-GAW training.
No pretrained models are included. Models are created only when you click Train Model.
The included sample is wholly synthetic; its forecasts and performance concern simulation
only. Nothing here establishes clinical accuracy or algorithmic novelty.

## Windows: quickest start

1. Extract the ZIP fully to a writable folder, for example Desktop/HealthForecast.
2. Install Python 3.11 or newer (3.11/3.12 recommended), including the Python launcher.
3. Double-click START_WINDOWS.bat. First launch requires internet to install packages.
4. Open http://127.0.0.1:8501 if the browser does not open automatically.
5. In module 1, upload data/sample_forecast_examples.csv, or the forecast_examples.csv
   from any ONE scenario in the earlier synthetic dataset package.
6. In module 2, leave 120 epochs / 3 folds initially and click Train Model.
7. In module 3, inspect actual measured results, enter measurements, and generate forecasts.
8. Download trained models and reports. Subsequent sessions can restore a saved run.

If the batch file cannot find py, install the Python launcher or use the commands below
with python instead of py -3. Do not run the application inside the compressed ZIP.

## Manual setup (PowerShell, Linux or macOS)

    python -m venv .venv

Windows (no PowerShell activation or execution-policy change required):

    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m streamlit run app.py

Linux/macOS:

    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m streamlit run app.py

Use --server.port 8502 if port 8501 is occupied. Initial dependency installation needs
internet; normal training and forecasting need no external API service or GPU.

## Module 1: Data Collection and Preparation

Accepts prepared forecast CSVs or sparse daily-record CSVs. Checks required columns,
numeric values, duplicate anchors, participant leakage, supplied target masks and dates.
It preserves supplied splits; otherwise assigns participants reproducibly to
60% train, 15% validation, 10% calibration, 15% test. It needs at least 20 participants
and at least two people with observed future Hb per horizon per split. The supplied
400-participant sample is the recommended starting point.

Prepared format: participant_id, anchor_date, horizon_days, hb0, weight0, hb_age_days,
weight_age_days, age, gender, steps_day_-4,...,steps_day_0, y_hb, y_weight, y_steps.
Optional split/is_synthetic/mask columns from the supplied dataset are supported.
Daily format: participant_id, date, hb, weight, steps, age, gender, optionally split
and is_synthetic. Missing measurements remain blank. Dates must be parseable.

Daily examples are anchored on measured Hb dates, with at least three recorded days
in a five-calendar-day step window. Hb and weight targets use the nearest observation
within +/-7 days of t+h. Steps target is a seven-day average ending at t+h with at
least five observed days. Targets are never imputed. Prepared CSVs must have been
built without future input leakage; supplied metadata is checked but arbitrary
external feature provenance cannot be independently proved by the app.

Both supported CSV types must contain only one simulation scenario. The four alternative
scenarios reuse people and are not independent cohorts to concatenate.

## Module 2: Joint Forecasting / HS-GAW

Implements the report's persistence references, training-only scaling of residuals,
24-unit tanh shared MLP, three linear output branches, masked participant-weighted L1
loss, and horizon-specific grid search with the stated standard-error gate. Analytic
NumPy gradients and actual Adam parameter updates are provided. This is the same MLP
mathematics as the report, implemented without a PyTorch dependency.

All preprocessing is fitted separately within each training fold. Gender is one-hot
encoded; all numeric inputs get missingness flags. All-missing input columns use zero
as the imputation value. Missing baseline weight uses its training median as the
persistence reference (zero only if the whole training column is absent). Future
target missingness never becomes an input feature. Train and validation people form
the grouped-CV development pool. Calibration and test people are excluded from tuning.

Each horizon tests (0,0), (.3,0), (0,.3), (.3,.3), (1,1) over 3 or 5 participant folds.
The best mean validation Hb participant-MAE candidate is accepted only if its mean
improvement over (0,0) exceeds SD(fold improvements)/sqrt(K), with sample SD (ddof=1).
Otherwise both weights are zero. This is a heuristic, not a statistical guarantee.
Models are refitted on development people only. No per-person gate operates at inference.

## Module 3: Evaluation and Dashboard

Compares persistence, Ridge, gradient boosting, Hb-only MLP, fixed (1,1) multitask and
HS-GAW. Same baseline inputs and participant splits are used; final MLP comparisons
share the seed, architecture, optimiser, learning rate and epoch budget. Ridge/boosting
use fixed documented defaults, not equal-budget hyperparameter searches. This package
does NOT implement GradNorm, uncertainty weighting or a shuffled-auxiliary control;
add these before claiming the report's full benchmark suite is completed.

Hb MAE is averaged per participant then across participants. RMSE is observation-weighted.
Skill = 1 - model participant-MAE / persistence participant-MAE. Paired improvement
is Hb-only MAE minus HS-GAW MAE, so positive means HS-GAW is better. Uses 2,000
participant bootstrap resamples for the 95% interval. Auxiliary outputs are evaluated
only when that task was active in training. If a gate switches an auxiliary off, the
dashboard displays a clearly labelled persistence fallback rather than an untrained head.
Approximate paired normal p-values with Holm correction across horizons are secondary;
these are not the gate rule and are not exact small-sample significance tests.

Prediction intervals use a separate participant calibration group: one score per
participant is their maximum absolute error across all that person's calibration
anchors. The 90% quantile uses rank ceil((n+1)*.9). Insufficient participants gives no
interval. Both anchor coverage and all-anchors-per-participant coverage are reported.
The cluster conformal interpretation requires exchangeable participants and comparable
anchor generation; drift or informative follow-up can invalidate coverage. It is not
an unconditional clinical guarantee. Hb and weight intervals are left unbounded;
steps predictions are clipped at zero. Formula details are in FORMULAS.md.

## Local saved files

Each completed run is stored under runs/<timestamp>/:

- horizon_30/model.npz and horizon_90/model.npz: neural weights, preprocessing, scales.
- metadata.json per horizon: selected weights, gate, calibration widths, settings.
- metrics.csv: measured test errors; cross_validation.csv: candidate fold scores.
- test_predictions.csv: actual and predicted held-out values and participant IDs.
- report.json: provenance SHA-256, settings, results and limitations.

Models load with allow_pickle=False. CSV uploads live in session memory and are not
automatically copied to disk. Exported test predictions contain uploaded participant IDs.
Keep identifiable data out of classroom demonstrations. Access is bound to localhost.

## Scientific scope

No result is prefilled or guaranteed. Zero selected weights and worse performance are
valid outcomes. Do not tune repeatedly on test results and call that set untouched.
For a stronger simulation paper, prespecify settings and evaluate multiple generator
seeds and network seeds, then add the missing controls/baselines above. Real linked
longitudinal data is needed for real-world Hb forecasting claims. A sample with same-day
Hb anchors does not validate older baseline Hb measurements or other patient populations.

## Verification

    python -m unittest discover -s tests -v

Tests include analytic-gradient checking, missing-target gradients, zero-weight
equivalence, gate behaviour, leakage rejection and complete training/serialization.
