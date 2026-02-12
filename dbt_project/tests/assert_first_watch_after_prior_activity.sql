/*
  Custom data test: assert_first_watch_after_prior_activity

  Core business rule validation for the training dataset.
  Ensures that every row in mart_training_dataset has a first_watch_at timestamp
  that is strictly greater than first_ever_activity_at.

  This guarantees that we only train on devices that had prior viewing history
  before adopting a new streaming service, which is essential for the prediction
  model to have meaningful features.

  A passing test returns zero rows.
*/

select
    smba_id,
    target_service,
    first_watch_at,
    first_ever_activity_at

from {{ ref('mart_training_dataset') }}
where first_watch_at <= first_ever_activity_at
