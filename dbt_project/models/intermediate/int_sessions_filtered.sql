{{
  config(
    materialized = 'table',
    partition_by = {
      "field": "exposure_date",
      "data_type": "date",
      "granularity": "day"
    }
  )
}}

/*
  Intermediate model: int_sessions_filtered

  Filters staged viewership data to meaningful sessions (> 60 seconds).
  Materialized as a partitioned table on exposure_date to support efficient
  date-range joins in downstream feature engineering.
*/

select *
from {{ ref('stg_streaming_viewership') }}
where duration_seconds > 60
