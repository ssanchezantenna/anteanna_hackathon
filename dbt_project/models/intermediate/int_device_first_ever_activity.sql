{{
  config(
    materialized = 'ephemeral'
  )
}}

/*
  Intermediate model: int_device_first_ever_activity

  Finds the earliest viewing session per device (smba_id) across ALL sources,
  not just target streaming services. This timestamp is used to ensure that a
  "first watch" on a target service is not the device's very first activity
  overall -- we need prior viewing history to build meaningful features.
*/

select
    smba_id,
    min(exposure_start_at) as first_ever_activity_at

from {{ ref('int_sessions_filtered') }}
group by smba_id
