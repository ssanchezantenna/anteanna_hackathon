-- =============================================================================
-- 01_data_profiling.sql
-- Purpose: Profile the raw Samba TV streaming viewership data to understand
--          volume, coverage, value distributions, and data quality before
--          building the first-watch prediction pipeline.
--
-- Source:   antenna-reporting.samba_tv.streaming_viewership_2_1_inc
-- Date range: Oct 2025 - Jan 2026
-- Run each query block independently in BigQuery console.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. HIGH-LEVEL SUMMARY
--    Row count, date range, and cardinality of key dimensions.
--    This establishes the scale of the dataset and confirms the expected
--    time window before any downstream modeling.
-- ---------------------------------------------------------------------------
SELECT
    COUNT(*)                                                     AS total_rows,
    COUNT(DISTINCT smba_id)                                      AS distinct_devices,
    COUNT(DISTINCT application)                                  AS distinct_applications,
    COUNT(DISTINCT title)                                        AS distinct_titles,
    MIN(TIMESTAMP_SECONDS(exposure_start_ts))                    AS earliest_session,
    MAX(TIMESTAMP_SECONDS(exposure_start_ts))                    AS latest_session,
    DATE_DIFF(
        DATE(MAX(TIMESTAMP_SECONDS(exposure_start_ts))),
        DATE(MIN(TIMESTAMP_SECONDS(exposure_start_ts))),
        DAY
    )                                                            AS date_span_days
FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
;


-- ---------------------------------------------------------------------------
-- 2. APPLICATION DISTRIBUTION
--    Exact spelling of every application value, with session and device counts.
--    Critical for identifying the precise strings that match each of the 8
--    target services (Amazon Prime Video, Apple TV+, Disney+, Max, Netflix,
--    Hulu, Peacock, Paramount+) and for discovering linear/other sources.
-- ---------------------------------------------------------------------------
SELECT
    application,
    COUNT(*)                    AS session_count,
    COUNT(DISTINCT smba_id)     AS device_count
FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
GROUP BY application
ORDER BY session_count DESC
;


-- ---------------------------------------------------------------------------
-- 3. SESSION DURATION DISTRIBUTION
--    Bucket sessions by computed duration to understand viewing quality.
--    Buckets:
--      <= 0 seconds   -- potential data issues (negative or zero duration)
--      1 - 60 seconds -- very short / accidental / channel surfing
--      61 - 300 s     -- short but intentional viewing
--      301 - 1800 s   -- standard viewing (5-30 min)
--      > 1800 s       -- long-form / binge viewing (30+ min)
--    The <=0 and 1-60s buckets inform the 60-second minimum filter used in
--    the first-watch definition.
-- ---------------------------------------------------------------------------
WITH sessions AS (
    SELECT
        exposure_end_ts - exposure_start_ts AS duration_seconds
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
)

SELECT
    CASE
        WHEN duration_seconds <= 0    THEN '1: <= 0s (invalid)'
        WHEN duration_seconds <= 60   THEN '2: 1-60s (very short)'
        WHEN duration_seconds <= 300  THEN '3: 61-300s (short)'
        WHEN duration_seconds <= 1800 THEN '4: 301-1800s (standard)'
        ELSE                               '5: > 1800s (long-form)'
    END                                    AS duration_bucket,
    COUNT(*)                               AS session_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
    MIN(duration_seconds)                  AS min_duration,
    MAX(duration_seconds)                  AS max_duration,
    ROUND(AVG(duration_seconds), 1)        AS avg_duration
FROM sessions
GROUP BY duration_bucket
ORDER BY duration_bucket
;


-- ---------------------------------------------------------------------------
-- 4. CONTENT TYPE DISTRIBUTION
--    Understand what content_type values exist and their prevalence.
--    Useful for building content-preference features later (movie vs series
--    vs other). NULL values are surfaced explicitly.
-- ---------------------------------------------------------------------------
SELECT
    IFNULL(content_type, '__NULL__')  AS content_type,
    COUNT(*)                          AS session_count,
    COUNT(DISTINCT smba_id)           AS device_count,
    COUNT(DISTINCT title)             AS distinct_titles,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_sessions
FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
GROUP BY content_type
ORDER BY session_count DESC
;


-- ---------------------------------------------------------------------------
-- 5. NETWORK COLUMN EXPLORATION
--    The network column likely identifies linear TV / cable channels vs
--    streaming apps. Understanding its relationship to application helps
--    distinguish streaming-native sessions from linear broadcasts, which
--    matters for the "prior activity on other services/networks" rule.
-- ---------------------------------------------------------------------------

-- 5a. Top network values by session count
SELECT
    IFNULL(network, '__NULL__')  AS network,
    COUNT(*)                     AS session_count,
    COUNT(DISTINCT smba_id)      AS device_count
FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
GROUP BY network
ORDER BY session_count DESC
LIMIT 50
;

-- 5b. Cross-tabulation: application vs network (top combinations)
--     Reveals whether streaming apps have NULL/empty network values
--     while linear content has populated network values.
SELECT
    application,
    IFNULL(network, '__NULL__') AS network,
    COUNT(*)                    AS session_count,
    COUNT(DISTINCT smba_id)     AS device_count
FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
GROUP BY application, network
ORDER BY session_count DESC
LIMIT 100
;

-- 5c. Devices that have BOTH linear (non-null network) and streaming sessions.
--     Quantifies the overlap -- these are devices we can build rich prior-
--     behavior features for.
WITH device_flags AS (
    SELECT
        smba_id,
        COUNTIF(network IS NOT NULL AND network != '') AS linear_sessions,
        COUNTIF(application IN (
            'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
            'Netflix', 'Hulu', 'Peacock', 'Paramount+'
        ))                                              AS streaming_sessions
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    GROUP BY smba_id
)

SELECT
    CASE
        WHEN linear_sessions > 0 AND streaming_sessions > 0 THEN 'both'
        WHEN linear_sessions > 0                             THEN 'linear_only'
        WHEN streaming_sessions > 0                          THEN 'streaming_only'
        ELSE                                                      'neither'
    END                     AS device_category,
    COUNT(*)                AS device_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_devices
FROM device_flags
GROUP BY device_category
ORDER BY device_count DESC
;


-- ---------------------------------------------------------------------------
-- 6. GEOGRAPHIC COVERAGE
--    Top 30 cities by distinct device count. Validates geographic diversity
--    and identifies dominant markets. Also checks NULL/empty city prevalence
--    to gauge how usable geography is as a feature.
-- ---------------------------------------------------------------------------
SELECT
    IFNULL(city, '__NULL__')     AS city,
    COUNT(DISTINCT smba_id)      AS device_count,
    COUNT(*)                     AS session_count,
    COUNT(DISTINCT application)  AS distinct_applications
FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
GROUP BY city
ORDER BY device_count DESC
LIMIT 30
;
