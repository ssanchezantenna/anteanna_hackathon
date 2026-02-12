{{
  config(
    materialized = 'ephemeral'
  )
}}

/*
  Intermediate model: int_valid_first_watch_events

  Joins first-watch-per-service events with first-ever-activity timestamps.
  Applies two critical filters:
    1. The first watch on a target service must occur AFTER the device's
       first-ever activity (ensuring prior viewing history exists).
    2. The first watch must be on or after 2025-11-01 to align with the
       project's date range for training and testing.
*/

with first_watch as (

    select * from {{ ref('int_device_service_first_session') }}

),

first_activity as (

    select * from {{ ref('int_device_first_ever_activity') }}

)

select
    fw.smba_id,
    fw.target_service,
    fw.first_watch_title,
    fw.first_watch_content_type,
    fw.city,
    fw.postal_code,
    fw.first_watch_at,
    fw.first_watch_date,
    fw.duration_seconds,
    fw.hour_of_day,
    fw.day_of_week,
    fa.first_ever_activity_at

from first_watch as fw
inner join first_activity as fa
    on fw.smba_id = fa.smba_id
where
    -- Core business rule: first watch must come after prior activity
    fw.first_watch_at > fa.first_ever_activity_at
    -- Date range filter for training/test windows
    and fw.first_watch_date >= '2025-11-01'
