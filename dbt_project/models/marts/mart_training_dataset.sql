{{
  config(
    materialized = 'table',
    partition_by = {
      "field": "first_watch_date",
      "data_type": "date",
      "granularity": "day"
    }
  )
}}

/*
  Mart model: mart_training_dataset

  Final training dataset for the "first watch" prediction model.
  Joins validated first-watch events with 30-day prior behavioral features.

  Includes:
    - Target label: target_service (the service the device adopted)
    - First-watch metadata: title, content type, timestamp, duration
    - Geographic features: city, postal_code
    - Prior behavioral features: service flags, activity metrics, temporal patterns
    - Data split column: 'train' for Nov-Dec 2025, 'test' for Jan 2026+
*/

with valid_events as (

    select * from {{ ref('int_valid_first_watch_events') }}

),

prior_features as (

    select * from {{ ref('int_device_prior_features') }}

),

joined as (

    select
        -- =====================================================================
        -- Identifiers
        -- =====================================================================
        ve.smba_id,
        ve.target_service,

        -- =====================================================================
        -- First-watch metadata
        -- =====================================================================
        ve.first_watch_title,
        ve.first_watch_content_type,
        ve.first_watch_at,
        ve.first_watch_date,
        ve.duration_seconds    as first_watch_duration_seconds,
        ve.hour_of_day         as first_watch_hour_of_day,
        ve.day_of_week         as first_watch_day_of_week,
        ve.first_ever_activity_at,

        -- =====================================================================
        -- Geographic features
        -- =====================================================================
        ve.city,
        ve.postal_code,

        -- =====================================================================
        -- Prior 30-day service flags
        -- =====================================================================
        pf.has_netflix,
        pf.has_amazon,
        pf.has_disney,
        pf.has_max,
        pf.has_apple_tv,
        pf.has_hulu,
        pf.has_peacock,
        pf.has_paramount,
        pf.has_linear,

        -- =====================================================================
        -- Prior 30-day activity metrics
        -- =====================================================================
        pf.active_days_30d,
        pf.total_duration_30d,
        pf.distinct_titles_30d,
        pf.total_sessions_30d,

        -- =====================================================================
        -- Prior 30-day service diversity
        -- =====================================================================
        pf.distinct_services_30d,

        -- =====================================================================
        -- Prior 30-day temporal patterns
        -- =====================================================================
        pf.avg_hour_of_day,
        pf.stddev_hour_of_day,
        pf.weekend_ratio,
        pf.primetime_ratio,

        -- =====================================================================
        -- Prior 30-day content preferences
        -- =====================================================================
        pf.dominant_content_type,
        pf.avg_session_duration,

        -- =====================================================================
        -- Data split: train (Nov-Dec 2025) vs test (Jan 2026+)
        -- =====================================================================
        case
            when ve.first_watch_date < '2026-01-01' then 'train'
            else 'test'
        end as data_split

    from valid_events as ve
    inner join prior_features as pf
        on  ve.smba_id        = pf.smba_id
        and ve.target_service = pf.target_service
        and ve.first_watch_at = pf.first_watch_at

)

select * from joined
