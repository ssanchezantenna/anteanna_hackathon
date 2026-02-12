# First Watch Prediction System -- Implementation Plan

## Goal
Predict the most likely first title a new streaming subscriber will watch, outputted as a probability distribution over titles (not just top-1). Uses BigQuery/dbt for feature engineering and Python (LightGBM) for modeling, exposed via a CLI tool.

## Target Services
`Amazon Prime Video`, `Apple TV+`, `Disney+`, `Max`, `Netflix`, `Hulu`, `Peacock`, `Paramount+`

## Source Data
```sql
-- BigQuery table: antenna-reporting.samba_tv.streaming_viewership_2_1_inc
-- Columns: smba_id, application, city, content_type, title, exposure_start_ts, exposure_end_ts, network, postal_code
-- Date range: Oct 2025 - Jan 2026
```

## Key Business Rule: "First Watch" Definition
A valid first watch must satisfy ALL of:
1. First session on a **target service** for a given `smba_id`
2. Session duration > 60 seconds (real viewing, not accidental)
3. The device **already had viewership on other services/networks** before this date (if it's the device's first-ever session across all sources, it's likely panel entry -- exclude it)
4. **The first watch date must be on or after November 1, 2025** (not in the first month of the dataset). This guarantees a full 30-day lookback window to confirm the device had no prior activity on that service, ensuring it's a genuine first watch and not a pre-existing subscriber whose earlier sessions fell before the dataset start date.

---

## Project Structure

```
antenna_hackathon/
├── README.md
├── PLAN.md
├── FEATURE_PROPOSALS.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── dbt_project/
│   ├── dbt_project.yml
│   ├── profiles.yml.example
│   ├── packages.yml
│   ├── models/
│   │   ├── sources.yml
│   │   ├── staging/
│   │   │   ├── _staging__models.yml
│   │   │   └── stg_streaming_viewership.sql
│   │   ├── intermediate/
│   │   │   ├── _intermediate__models.yml
│   │   │   ├── int_sessions_filtered.sql
│   │   │   ├── int_device_service_first_session.sql
│   │   │   ├── int_device_first_ever_activity.sql
│   │   │   ├── int_valid_first_watch_events.sql
│   │   │   └── int_device_prior_features.sql
│   │   └── marts/
│   │       ├── _marts__models.yml
│   │       ├── mart_training_dataset.sql
│   │       └── mart_title_popularity.sql
│   ├── macros/
│   │   └── target_services.sql
│   └── tests/
│       └── assert_first_watch_after_prior_activity.sql
├── exploration/
│   ├── 01_data_profiling.sql
│   ├── 02_first_watch_validation.sql
│   └── 03_feature_distributions.sql
├── first_watch_model/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── baseline.py
│   ├── classifier.py
│   ├── ensemble.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── cli/
│   ├── __init__.py
│   └── main.py
├── models/                    # Saved model artifacts (gitignored)
│   └── .gitkeep
├── examples/
│   ├── sample_input.csv
│   └── sample_output.csv
└── tests/
    ├── test_features.py
    ├── test_baseline.py
    ├── test_classifier.py
    ├── test_ensemble.py
    └── test_cli.py
```

---

## Phase 1: Data Exploration Queries (`exploration/`)

### `01_data_profiling.sql`
- Row counts, date range, distinct devices/applications/titles
- Application distribution (exact service name spellings, session counts, device counts)
- Session duration distribution (buckets: <=0, 1-60s, 61-300s, 301-1800s, >1800s)
- Content type distribution
- Network column exploration (understand linear TV identification)
- Geographic coverage (top cities by device count)

### `02_first_watch_validation.sql`
- Preview the first-watch identification logic: count valid first watches per service
- Title distribution of first watches per service (top 20) -- validates that results are sensible

### `03_feature_distributions.sql`
- Distribution of service counts per device in 30-day windows
- Activity level distributions (sessions/day, duration/day)
- Content type preference distributions

---

## Phase 2: dbt Models

### Macro: `target_services.sql`
Returns the 8 target service names as a SQL IN-list and as a Jinja list.

### Staging: `stg_streaming_viewership.sql`
- Cast `exposure_start_ts`/`exposure_end_ts` from unix to TIMESTAMP
- Compute `duration_seconds`
- Flag `is_target_service` (boolean)
- Extract `exposure_date`, `hour_of_day`, `day_of_week`
- Filter: duration > 0, non-null smba_id and application
- Materialized as **view**

### Intermediate Models

**`int_sessions_filtered.sql`** (materialized as **table**, partitioned by `exposure_start_at`)
- Filter to sessions > 60 seconds from staging

**`int_device_service_first_session.sql`** (ephemeral)
- For each (smba_id, target_service), find the earliest session using `ROW_NUMBER()`
- Capture: first_watch_title, first_watch_content_type, city, postal_code, first_watch_at, first_watch_date, duration, hour, day_of_week

**`int_device_first_ever_activity.sql`** (ephemeral)
- `MIN(exposure_start_at)` per smba_id across ALL sources (not just target services)

**`int_valid_first_watch_events.sql`** (ephemeral)
- JOIN first-per-service with first-ever-activity
- **Keep only rows where `first_watch_at > first_ever_activity_at`** (the device was already active before this first watch)
- **Filter: `first_watch_date >= '2025-11-01'`** (exclude October -- the first month is a lookback-only window to guarantee we can confirm it's truly a first watch)

**`int_device_prior_features.sql`** (ephemeral)
- For each valid first-watch event, join back to `int_sessions_filtered` for sessions in the **30 days before** the first watch
- Compute features:

| Feature Group | Features |
|---|---|
| **Service flags** | `has_netflix`, `has_amazon`, `has_disney`, `has_max`, `has_apple_tv`, `has_hulu`, `has_peacock`, `has_paramount`, `has_linear` |
| **Activity metrics** | `active_days_30d`, `total_duration_30d`, `distinct_titles_30d`, `total_sessions_30d` |
| **Service diversity** | `distinct_services_30d` |
| **Temporal patterns** | `avg_hour_of_day`, `stddev_hour_of_day`, `weekend_ratio`, `primetime_ratio` |
| **Content preferences** | `dominant_content_type`, `avg_session_duration` |

### Marts

**`mart_training_dataset.sql`** (table, partitioned by `first_watch_date`)
- JOIN `int_valid_first_watch_events` with `int_device_prior_features`
- Include `city`, `postal_code` as geographic features
- Add `data_split` column: `'train'` for Nov-Dec 2025, `'test'` for Jan 2026 (October is excluded -- used only as lookback window)

**`mart_title_popularity.sql`** (table)
- Empirical P(title | service) distribution: count first watches per (service, title), compute `empirical_probability`
- Include `popularity_rank` per service

---

## Phase 3: Python Model

### Architecture: One LightGBM classifier per service + empirical blending

**Why per-service?** Each service has a completely different title catalog. A Disney+ title can never be the first watch on Netflix.

### `config.py`
- Constants: service list, BQ references, feature column names
- Hyperparameters: `MAX_TITLES_PER_SERVICE=75`, `SMOOTHING_ALPHA=0.3`, `TEMPERATURE=1.5`, LightGBM params

### `data_loader.py`
- `load_training_data(split)` -- query `mart_training_dataset` from BQ
- `load_title_popularity()` -- query `mart_title_popularity` from BQ

### `features.py`
- `build_title_encoder(df, service)` -- top-N titles + `__OTHER__` bucket for rare titles
- `prepare_features(df, encoders, fit)` -- fill nulls, ordinal-encode categoricals, return feature matrix

### `baseline.py` -- `PopularityBaseline`
- Stores empirical P(title | service) per service
- Prediction = same distribution for every subscriber of a service (floor benchmark)

### `classifier.py` -- `ServiceClassifier`
- LightGBM multi-class classifier (one per service)
- Target classes: top-75 titles + `__OTHER__`
- **Temperature scaling** on `predict_proba`: `P_scaled(i) = softmax(logit_i / T)`
  - T > 1 = flatter/more diverse distribution (avoids over-predicting #1 title)
  - T = 1 = raw model output

### `ensemble.py` -- `EnsemblePredictor`
- **Blending formula**: `P_final = (1 - alpha) * P_classifier + alpha * P_empirical`
- `alpha = 0.3` ensures ~30% of the prediction comes from the base rate
- This prevents degenerate predictions where everyone gets the same top title
- Falls back to pure popularity baseline for services with insufficient training data

### `train.py`
1. Load training data from BQ (split='train')
2. Prepare features (fit encoders)
3. Train `PopularityBaseline`
4. Train one `ServiceClassifier` per service (skip if <50 samples)
5. Build `EnsemblePredictor`
6. Save all artifacts to `models/` via joblib

### `predict.py`
1. Load saved ensemble + feature encoders
2. Accept a DataFrame of subscribers with service + optional feature columns
3. Fill missing features with defaults (0 for numeric, "__UNKNOWN__" for categorical)
4. Run ensemble prediction per service
5. Return DataFrame: `subscriber_id, service, rank, title, probability`

### `evaluate.py`
Metrics (all computed per-service and macro-averaged):

| Metric | Purpose |
|---|---|
| **Top-k accuracy** (k=1,3,5,10) | Is the true title in the top-k predictions? |
| **MRR** | Average 1/rank of correct title |
| **Log-likelihood** | Calibration of probability estimates |
| **KL divergence** | Does aggregate prediction distribution match actual distribution? |

**Evaluation protocol**: Train on Nov-Dec 2025, test on Jan 2026 (temporal split; Oct is lookback-only). Compare: Baseline vs Classifier-only vs Ensemble.

---

## Phase 4: CLI Tool (`cli/main.py`)

Click-based CLI with three commands:

```bash
# Train the model (requires BigQuery access)
first-watch train

# Generate predictions from a CSV
first-watch predict --input examples/sample_input.csv --output predictions.csv --top-k 10 --format long

# Evaluate on test set
first-watch evaluate --detailed
```

**Input CSV format** (only `service` is required; features are optional):
```csv
service,has_netflix,has_amazon,...,city
Netflix,0,1,...,New York
```

**Output CSV format** (long):
```csv
subscriber_id,service,rank,title,probability
0,Netflix,1,Squid Game,0.082
0,Netflix,2,Wednesday,0.068
```

---

## Phase 5: Examples & Tests

### `examples/sample_input.csv`
5 sample subscribers across different services with realistic feature values.

### `examples/sample_output.csv`
Expected output showing top-10 title predictions with probabilities per subscriber.

### Unit tests (`tests/`)
- `test_features.py` -- null handling, encoding, output shape
- `test_baseline.py` -- probabilities sum to 1.0, sorting
- `test_classifier.py` -- predict_proba shape, temperature scaling
- `test_ensemble.py` -- valid blended distributions, baseline fallback
- `test_cli.py` -- Click CliRunner smoke tests

---

## Implementation Order

0. **Save this plan**: This plan is saved as `PLAN.md` in the repo root
1. **Scaffold**: Directory structure, `pyproject.toml`, `.gitignore`, dbt config files
2. **Exploration SQL**: Write and document exploration queries
3. **Feature Discovery & Proposal**: After running the exploration queries, analyze the data to propose additional features beyond the ones listed above. Investigate:
   - **Title-level signals**: Are there titles that act as "gateway" content for a service? (e.g., people who watched X on Netflix are more likely to try Disney+)
   - **Recency patterns**: Days since last session, trend in viewing frequency (increasing/decreasing)
   - **Service adoption order**: Which services the device adopted first -- does the order predict what they watch on the next service?
   - **Content type affinity**: Ratio of movies vs series vs other content types in prior viewing
   - **Binge behavior**: Max consecutive days active, longest single session
   - **Co-viewing signals**: Number of concurrent sessions (proxy for household size/engagement)
   - **Geographic title affinity**: Do certain cities/regions have stronger preferences for specific titles?
   - Document proposed features with rationale in a `FEATURE_PROPOSALS.md` file, then update dbt models to include the approved ones
4. **dbt models**: staging -> intermediate -> marts, run `dbt build && dbt test`
5. **Python model**: config -> data_loader -> features -> baseline -> classifier -> ensemble -> train -> predict
6. **Evaluation**: Implement evaluate.py, run on test set, tune alpha/temperature
7. **CLI**: Implement Click commands, test with sample input
8. **Examples & tests**: Sample files, unit tests, README update

---

## Key Design Decisions

- **Per-service classifiers** avoid cross-contamination of title catalogs between services
- **Temperature scaling (T=1.5)** flattens the distribution to prevent over-predicting the top title
- **Empirical blending (alpha=0.3)** ensures output resembles real viewership distribution
- **Temporal train/test split** (Nov-Dec train, Jan test; Oct is lookback-only) simulates real-world prediction scenario
- **First-month exclusion** (October) guarantees a full 30-day lookback to confirm first watches are genuine
- **Top-75 titles + __OTHER__** keeps class space manageable while covering majority of first watches
