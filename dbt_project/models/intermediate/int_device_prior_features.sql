{{
  config(
    materialized = 'ephemeral'
  )
}}

/*
  Intermediate model: int_device_prior_features

  For each valid first-watch event, computes behavioral features from the
  30-day window of viewing activity PRIOR to the first watch on the target service.

  Feature groups:
    - Service flags: binary indicators for each target service + linear TV
    - Activity metrics: active days, total duration, distinct titles, session count
    - Service diversity: count of distinct services used
    - Temporal patterns: avg/stddev hour, weekend ratio, primetime ratio
    - Content preferences: dominant content type, average session duration
*/

with valid_events as (

    select * from {{ ref('int_valid_first_watch_events') }}

),

prior_sessions as (

    /*
      Join each valid first-watch event to ALL sessions for that device
      in the 30 days leading up to (but not including) the first watch moment.
    */
    select
        ve.smba_id,
        ve.target_service,
        ve.first_watch_at,
        s.application,
        s.title,
        s.content_type,
        s.network,
        s.duration_seconds,
        s.exposure_date,
        s.hour_of_day,
        s.day_of_week,
        s.exposure_start_at

    from valid_events as ve
    inner join {{ ref('int_sessions_filtered') }} as s
        on ve.smba_id = s.smba_id
        and s.exposure_start_at < ve.first_watch_at
        and s.exposure_date >= date_sub(ve.first_watch_date, interval 30 day)

),

aggregated as (

    select
        smba_id,
        target_service,
        first_watch_at,

        -- =====================================================================
        -- Service flags: has the device used each service in the prior 30 days?
        -- =====================================================================
        max(case when application = 'Netflix'            then 1 else 0 end) as has_netflix,
        max(case when application = 'Amazon Prime Video' then 1 else 0 end) as has_amazon,
        max(case when application = 'Disney+'            then 1 else 0 end) as has_disney,
        max(case when application = 'Max'                then 1 else 0 end) as has_max,
        max(case when application = 'Apple TV+'          then 1 else 0 end) as has_apple_tv,
        max(case when application = 'Hulu'               then 1 else 0 end) as has_hulu,
        max(case when application = 'Peacock'            then 1 else 0 end) as has_peacock,
        max(case when application = 'Paramount+'         then 1 else 0 end) as has_paramount,
        max(case when application not in {{ target_services() }} then 1 else 0 end) as has_linear,

        -- =====================================================================
        -- Activity metrics
        -- =====================================================================
        count(distinct exposure_date)  as active_days_30d,
        sum(duration_seconds)          as total_duration_30d,
        count(distinct title)          as distinct_titles_30d,
        count(*)                       as total_sessions_30d,

        -- =====================================================================
        -- Service diversity
        -- =====================================================================
        count(distinct application)    as distinct_services_30d,

        -- =====================================================================
        -- Temporal patterns
        -- =====================================================================
        avg(hour_of_day)               as avg_hour_of_day,
        stddev(hour_of_day)            as stddev_hour_of_day,

        -- Weekend ratio: fraction of sessions on Sat (7) or Sun (1)
        safe_divide(
            countif(day_of_week in (1, 7)),
            count(*)
        ) as weekend_ratio,

        -- Primetime ratio: fraction of sessions between 7 PM and 11 PM
        safe_divide(
            countif(hour_of_day between 19 and 22),
            count(*)
        ) as primetime_ratio,

        -- =====================================================================
        -- Content preferences
        -- =====================================================================
        approx_top_count(content_type, 1)[offset(0)].value as dominant_content_type,

        avg(duration_seconds)          as avg_session_duration

    from prior_sessions
    group by smba_id, target_service, first_watch_at

)

select * from aggregated
