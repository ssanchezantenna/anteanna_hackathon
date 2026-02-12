{{
  config(
    materialized = 'view'
  )
}}

/*
  Staging model: stg_streaming_viewership

  Cleans and enriches raw Samba TV streaming viewership data.
  - Casts unix timestamps to TIMESTAMP
  - Computes session duration in seconds
  - Flags whether the application is one of the 8 target streaming services
  - Extracts date, hour, and day-of-week for downstream temporal analysis
  - Filters out invalid rows (null device/app, zero or negative duration)
*/

with source as (

    select *
    from {{ source('samba_tv', 'streaming_viewership_2_1_inc') }}

),

cleaned as (

    select
        -- Device and content identifiers
        smba_id,
        application,
        title,
        content_type,
        network,

        -- Geography
        city,
        postal_code,

        -- Timestamps: cast from unix epoch to BigQuery TIMESTAMP
        timestamp_seconds(cast(exposure_start_ts as int64)) as exposure_start_at,
        timestamp_seconds(cast(exposure_end_ts as int64))   as exposure_end_at,

        -- Duration in seconds
        cast(exposure_end_ts as int64) - cast(exposure_start_ts as int64) as duration_seconds,

        -- Target service flag
        application in {{ target_services() }} as is_target_service,

        -- Temporal features
        date(timestamp_seconds(cast(exposure_start_ts as int64))) as exposure_date,
        extract(hour from timestamp_seconds(cast(exposure_start_ts as int64)))      as hour_of_day,
        extract(dayofweek from timestamp_seconds(cast(exposure_start_ts as int64))) as day_of_week

    from source
    where
        -- Require valid device and application
        smba_id is not null
        and application is not null
        -- Require positive duration
        and cast(exposure_end_ts as int64) - cast(exposure_start_ts as int64) > 0

)

select * from cleaned
