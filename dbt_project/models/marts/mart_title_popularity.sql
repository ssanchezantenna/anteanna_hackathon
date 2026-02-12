{{
  config(
    materialized = 'table'
  )
}}

/*
  Mart model: mart_title_popularity

  Computes the empirical probability distribution P(title | service) based on
  first-watch events. For each target service, counts how many devices had that
  title as their first watch, then normalizes to an empirical probability.

  Use cases:
    - Prior probability input for Bayesian title prediction
    - Popularity-based ranking and analysis per service
*/

with first_watch_counts as (

    select
        target_service,
        first_watch_title,
        count(*) as first_watch_count

    from {{ ref('int_valid_first_watch_events') }}
    group by target_service, first_watch_title

),

service_totals as (

    select
        target_service,
        sum(first_watch_count) as total_first_watches

    from first_watch_counts
    group by target_service

),

with_probability as (

    select
        fwc.target_service,
        fwc.first_watch_title,
        fwc.first_watch_count,
        st.total_first_watches,

        -- Empirical probability: P(title | service)
        safe_divide(fwc.first_watch_count, st.total_first_watches) as empirical_probability,

        -- Popularity rank within each service (1 = most popular)
        row_number() over (
            partition by fwc.target_service
            order by fwc.first_watch_count desc
        ) as popularity_rank

    from first_watch_counts as fwc
    inner join service_totals as st
        on fwc.target_service = st.target_service

)

select * from with_probability
order by target_service, popularity_rank
