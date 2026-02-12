# anteanna_hackathon
  ---             
  What This Repo Does                                                                                                                            
                  
  "First Watch" is a prediction system that answers the question: "When a person subscribes to a new streaming service, what will be the first   
  title they watch?"

  It targets 8 major streaming services: Netflix, Amazon Prime Video, Apple TV+, Disney+, Max, Hulu, Peacock, and Paramount+.

  Data Source

  The system uses Samba TV viewership data from BigQuery (antenna-reporting.samba_tv.streaming_viewership_2_1_inc) covering Oct 2025 - Jan 2026.
  Each row represents a viewing session with device IDs, app names, titles, timestamps, and location.

  "First Watch" Definition

  A valid first watch must satisfy all of:
  1. It's the device's first-ever session on a specific target service
  2. Session > 60 seconds (not accidental)
  3. The device already had viewing activity on other services before (otherwise it's likely panel entry noise)
  4. The first watch date is on or after Nov 1, 2025 (guaranteeing a full 30-day lookback window)

  How the Model Works

  It's a 3-layer architecture:

  1. Popularity Baseline — empirical P(title | service): "what do most people watch first on Netflix?" Same answer for everyone on a service.
  This is the floor benchmark.
  2. LightGBM Classifier (one per service) — personalized multi-class classifier trained on subscriber features. Outputs a probability
  distribution over the top-75 titles + an __OTHER__ bucket. Uses temperature scaling (T=1.5) to flatten predictions so it doesn't just always
  predict the #1 title.
  3. Ensemble — blends the two: P_final = 0.7 * P_classifier + 0.3 * P_empirical. This keeps predictions personalized but grounded in real
  viewership patterns.

  Features Used

  For each subscriber approaching a new service, the model looks at their prior 30 days of viewing across all other services:
  ┌───────────────┬───────────────────────────────────────────────────────────────────────┐
  │   Category    │                               Features                                │
  ├───────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Service flags │ Which other services they already use (has_netflix, has_amazon, etc.) │
  ├───────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Activity      │ Active days, total viewing duration, distinct titles, total sessions  │
  ├───────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Temporal      │ Average hour of day, weekend ratio, primetime ratio                   │
  ├───────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Content       │ Dominant content type (series/movie), avg session duration            │
  ├───────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Geography     │ City                                                                  │
  ├───────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Signup timing │ Month, day of week, week of year                                      │
  └───────────────┴───────────────────────────────────────────────────────────────────────┘
  ---
  The Examples Explained

  examples/sample_input.csv — 5 hypothetical subscribers

  Each row is a person about to watch their first title on a particular service:
  #: 0
  Service: Netflix
  Profile Summary: Heavy viewer (22 active days, 45 sessions), uses Amazon+Disney+linear TV, New York, mostly series, watches late evening (20:30

    avg)
  ────────────────────────────────────────
  #: 1
  Service: Disney+
  Profile Summary: Moderate (15 days, 28 sessions), uses Netflix+Amazon+Apple TV, LA, prefers movies, watches earlier (18:00)
  ────────────────────────────────────────
  #: 2
  Service: Max
  Profile Summary: Very active (25 days, 60 sessions), uses Netflix+Hulu+linear, Chicago, series watcher, heavy primetime (70%)
  ────────────────────────────────────────
  #: 3
  Service: Amazon Prime
  Profile Summary: Light viewer (10 days, 15 sessions), uses Disney+Max+Peacock, Houston, movie preference
  ────────────────────────────────────────
  #: 4
  Service: Apple TV+
  Profile Summary: Power user (28 days, 70 sessions, 5 services), uses Netflix+Amazon+Disney+Paramount+linear, San Francisco, series
  examples/sample_output.csv — the predictions

  For each subscriber, the model outputs top-10 titles ranked by probability. Some highlights:

  - Subscriber 0 (Netflix): Top prediction is "A House of Dynamite" (28.5%), then "Stranger Things" (13.7%), "KPop Demon Hunters" (10.1%).
  Probabilities are spread out — the model is uncertain but directional.
  - Subscriber 1 (Disney+): "Ancient Aliens" dominates at 86.2%. The model is very confident. Remaining titles ("Fixer Upper", "Bluey", "The
  Simpsons") have tiny probabilities.
  - Subscriber 2 (Max): "The Big Bang Theory" at 92.7% — extremely high confidence. "Friends" is a distant second at 1.7%.
  - Subscriber 3 (Amazon Prime): "The Polar Express" leads at 27%, then "Judy Justice" (10.9%), "Magnum P.I." (10.6%). More evenly spread.
  - Subscriber 4 (Apple TV+): "A Charlie Brown Thanksgiving" at 90.4% — very confident. Then "Pluribus" (3.3%) and "The Morning Show" (1.1%).

  Output Format

  subscriber_id, service, rank, title, probability

  Each subscriber gets 10 rows (one per predicted title), sorted by descending probability. All probabilities per subscriber sum to ~1.0 after
  re-normalization.

  ---
  CLI Interface

  The system is exposed as a CLI tool called first-watch:

  first-watch train              # Train model (needs BigQuery)
  first-watch predict --input examples/sample_input.csv --output predictions.csv --top-k 10
  first-watch evaluate --detailed # Evaluate on Jan 2026 test set