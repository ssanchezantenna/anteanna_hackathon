{{
  config(
    materialized = 'ephemeral'
  )
}}

/*
  Intermediate model: int_device_service_first_session

  For each (smba_id, target_service) pair, identifies the earliest viewing session.
  Uses ROW_NUMBER() partitioned by device and application, ordered by session start time.
  Captures the first-watch metadata needed for the training dataset:
  title, content type, geography, timestamp, date, duration, and temporal features.
*/

with ranked_sessions as (

    select
        smba_id,
        application as target_service,
        title       as first_watch_title,
        content_type as first_watch_content_type,
        city,
        postal_code,
        exposure_start_at as first_watch_at,
        exposure_date     as first_watch_date,
        duration_seconds  as duration_seconds,
        hour_of_day,
        day_of_week,

        row_number() over (
            partition by smba_id, application
            order by exposure_start_at asc
        ) as session_rank

    from {{ ref('int_sessions_filtered') }}
    where is_target_service = true

)

select
    smba_id,
    target_service,
    first_watch_title,
    first_watch_content_type,
    city,
    postal_code,
    first_watch_at,
    first_watch_date,
    duration_seconds,
    hour_of_day,
    day_of_week

from ranked_sessions
where session_rank = 1
