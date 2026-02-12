-- =============================================================================
-- 02_first_watch_validation.sql
-- Purpose: Validate the first-watch identification logic before implementing
--          it in dbt. A valid first watch must satisfy ALL of:
--
--          1. First session on a TARGET SERVICE for a given smba_id
--          2. Session duration > 60 seconds (real viewing, not accidental)
--          3. The device already had viewership on OTHER services/networks
--             before this date (excludes panel-entry devices)
--          4. The first watch date is on or after November 1, 2025
--             (guarantees a full 30-day lookback to confirm it is truly
--             a first watch and not a pre-existing subscriber whose earlier
--             sessions fell before the dataset start date)
--
-- Source:   antenna-reporting.samba_tv.streaming_viewership_2_1_inc
-- Run each query block independently in BigQuery console.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. VALID FIRST WATCHES PER SERVICE
--    Walks through the full identification logic step by step using CTEs,
--    then counts how many valid first-watch events each target service has.
--    This is the core number that determines training set size per service.
-- ---------------------------------------------------------------------------
WITH target_services AS (
    -- Explicit list of the 8 target streaming services
    SELECT service
    FROM UNNEST([
        'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
        'Netflix', 'Hulu', 'Peacock', 'Paramount+'
    ]) AS service
),

filtered_sessions AS (
    -- Base filter: only sessions longer than 60 seconds on target services.
    -- This satisfies Rule 2 (duration > 60s) and restricts to target apps.
    SELECT
        smba_id,
        application,
        title,
        content_type,
        city,
        postal_code,
        TIMESTAMP_SECONDS(exposure_start_ts)                AS exposure_start_at,
        DATE(TIMESTAMP_SECONDS(exposure_start_ts))          AS exposure_date,
        exposure_end_ts - exposure_start_ts                  AS duration_seconds
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE application IN (
        'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
        'Netflix', 'Hulu', 'Peacock', 'Paramount+'
    )
      AND (exposure_end_ts - exposure_start_ts) > 60
),

first_per_service AS (
    -- Rule 1: For each (device, service), find the earliest qualifying session.
    -- ROW_NUMBER partitioned by device+service, ordered by timestamp, picks
    -- the very first session.
    SELECT
        smba_id,
        application                       AS service,
        title                             AS first_watch_title,
        content_type                      AS first_watch_content_type,
        city,
        postal_code,
        exposure_start_at                 AS first_watch_at,
        exposure_date                     AS first_watch_date,
        duration_seconds
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY smba_id, application
                ORDER BY exposure_start_at ASC
            ) AS rn
        FROM filtered_sessions
    )
    WHERE rn = 1
),

device_first_ever_activity AS (
    -- Find the earliest session across ALL sources (streaming + linear)
    -- for every device in the entire dataset. This is used to check Rule 3:
    -- the device must have had activity before the candidate first watch.
    SELECT
        smba_id,
        MIN(TIMESTAMP_SECONDS(exposure_start_ts)) AS first_ever_activity_at
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE (exposure_end_ts - exposure_start_ts) > 60
    GROUP BY smba_id
),

valid_first_watches AS (
    -- Apply Rules 3 and 4:
    --   Rule 3: first_watch_at > first_ever_activity_at
    --           (device was active before this first watch)
    --   Rule 4: first_watch_date >= '2025-11-01'
    --           (full 30-day lookback guaranteed)
    SELECT
        fps.*,
        dfea.first_ever_activity_at
    FROM first_per_service fps
    INNER JOIN device_first_ever_activity dfea
        ON fps.smba_id = dfea.smba_id
    WHERE fps.first_watch_at > dfea.first_ever_activity_at
      AND fps.first_watch_date >= '2025-11-01'
)

-- Final output: count of valid first watches per service
SELECT
    ts.service,
    COUNT(vfw.smba_id)          AS valid_first_watch_count,
    COUNT(DISTINCT vfw.smba_id) AS distinct_devices
FROM target_services ts
LEFT JOIN valid_first_watches vfw
    ON ts.service = vfw.service
GROUP BY ts.service
ORDER BY valid_first_watch_count DESC
;


-- ---------------------------------------------------------------------------
-- 2. TITLE DISTRIBUTION OF FIRST WATCHES PER SERVICE (TOP 20)
--    For each target service, show the 20 most common first-watch titles.
--    This is a sanity check: the top titles should be recognizable hits
--    (e.g., Squid Game on Netflix, Moana 2 on Disney+). If the top titles
--    look nonsensical, the identification logic may have a bug.
-- ---------------------------------------------------------------------------
WITH filtered_sessions AS (
    SELECT
        smba_id,
        application,
        title,
        content_type,
        TIMESTAMP_SECONDS(exposure_start_ts)       AS exposure_start_at,
        DATE(TIMESTAMP_SECONDS(exposure_start_ts)) AS exposure_date,
        exposure_end_ts - exposure_start_ts         AS duration_seconds
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE application IN (
        'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
        'Netflix', 'Hulu', 'Peacock', 'Paramount+'
    )
      AND (exposure_end_ts - exposure_start_ts) > 60
),

first_per_service AS (
    SELECT
        smba_id,
        application                       AS service,
        title                             AS first_watch_title,
        content_type                      AS first_watch_content_type,
        exposure_start_at                 AS first_watch_at,
        exposure_date                     AS first_watch_date,
        duration_seconds
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY smba_id, application
                ORDER BY exposure_start_at ASC
            ) AS rn
        FROM filtered_sessions
    )
    WHERE rn = 1
),

device_first_ever_activity AS (
    SELECT
        smba_id,
        MIN(TIMESTAMP_SECONDS(exposure_start_ts)) AS first_ever_activity_at
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE (exposure_end_ts - exposure_start_ts) > 60
    GROUP BY smba_id
),

valid_first_watches AS (
    SELECT fps.*
    FROM first_per_service fps
    INNER JOIN device_first_ever_activity dfea
        ON fps.smba_id = dfea.smba_id
    WHERE fps.first_watch_at > dfea.first_ever_activity_at
      AND fps.first_watch_date >= '2025-11-01'
),

ranked_titles AS (
    -- Rank titles within each service by first-watch frequency.
    -- Include the content type to see whether first watches skew
    -- toward movies, series, or other content.
    SELECT
        service,
        first_watch_title,
        first_watch_content_type,
        COUNT(*) AS first_watch_count,
        ROW_NUMBER() OVER (
            PARTITION BY service
            ORDER BY COUNT(*) DESC
        ) AS title_rank
    FROM valid_first_watches
    GROUP BY service, first_watch_title, first_watch_content_type
)

SELECT
    service,
    title_rank,
    first_watch_title,
    first_watch_content_type,
    first_watch_count,
    ROUND(
        first_watch_count * 100.0
        / SUM(first_watch_count) OVER (PARTITION BY service),
        2
    ) AS pct_of_service_first_watches
FROM ranked_titles
WHERE title_rank <= 20
ORDER BY service, title_rank
;
