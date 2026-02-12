{{
  config(
    materialized = 'table'
  )
}}

/*
  Mart model: mart_title_popularity

  Computes time-windowed empirical probability distributions:
    - P(title | service, month) -- monthly distribution for time-aware predictions
    - P(title | service)        -- global distribution as fallback

  The signup_month column allows the ML model to use the popularity baseline
  that matches the actual signup date, capturing shifts in title popularity
  over time (e.g. new show launches, seasonal content).
*/

with first_watch_with_month as (

    select
        target_service,
        first_watch_title,
        extract(MONTH from first_watch_date) as signup_month

    from {{ ref('int_valid_first_watch_events') }}

),

-- =========================================================================
-- Monthly distributions: P(title | service, month)
-- =========================================================================

monthly_counts as (

    select
        target_service,
        signup_month,
        first_watch_title,
        count(*) as first_watch_count

    from first_watch_with_month
    group by target_service, signup_month, first_watch_title

),

monthly_totals as (

    select
        target_service,
        signup_month,
        sum(first_watch_count) as total_first_watches

    from monthly_counts
    group by target_service, signup_month

),

monthly_probability as (

    select
        mc.target_service,
        mc.signup_month,
        mc.first_watch_title,
        mc.first_watch_count,
        mt.total_first_watches,

        safe_divide(mc.first_watch_count, mt.total_first_watches) as empirical_probability,

        row_number() over (
            partition by mc.target_service, mc.signup_month
            order by mc.first_watch_count desc
        ) as popularity_rank

    from monthly_counts as mc
    inner join monthly_totals as mt
        on mc.target_service = mt.target_service
        and mc.signup_month = mt.signup_month

),

-- =========================================================================
-- Global distributions: P(title | service) -- fallback when month unknown
-- =========================================================================

global_counts as (

    select
        target_service,
        first_watch_title,
        count(*) as first_watch_count

    from first_watch_with_month
    group by target_service, first_watch_title

),

global_totals as (

    select
        target_service,
        sum(first_watch_count) as total_first_watches

    from global_counts
    group by target_service

),

global_probability as (

    select
        gc.target_service,
        cast(null as INT64) as signup_month,
        gc.first_watch_title,
        gc.first_watch_count,
        gt.total_first_watches,

        safe_divide(gc.first_watch_count, gt.total_first_watches) as empirical_probability,

        row_number() over (
            partition by gc.target_service
            order by gc.first_watch_count desc
        ) as popularity_rank

    from global_counts as gc
    inner join global_totals as gt
        on gc.target_service = gt.target_service

),

-- =========================================================================
-- Union monthly + global rows
-- =========================================================================

combined as (

    select * from monthly_probability
    union all
    select * from global_probability

)

select * from combined
order by target_service, signup_month nulls last, popularity_rank
