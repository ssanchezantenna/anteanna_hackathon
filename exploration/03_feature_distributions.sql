-- =============================================================================
-- 03_feature_distributions.sql
-- Purpose: Explore the distributions of candidate features that will feed
--          the first-watch prediction model. Understanding these distributions
--          informs feature engineering decisions: binning thresholds, outlier
--          handling, normalization strategies, and whether a feature has
--          enough variance to be useful.
--
-- All features are computed over a 30-day lookback window anchored to each
-- device's valid first-watch date, mirroring the exact feature construction
-- that will happen in the dbt intermediate layer.
--
-- Source:   antenna-reporting.samba_tv.streaming_viewership_2_1_inc
-- Run each query block independently in BigQuery console.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. DISTRIBUTION OF DISTINCT SERVICE COUNTS PER DEVICE (30-DAY WINDOW)
--    How many different streaming services does a device use in the 30 days
--    before their first watch on a new service? This measures "service
--    diversity" -- a key predictor of cross-platform adoption behavior.
--    Devices using many services may have different title preferences than
--    single-service households.
-- ---------------------------------------------------------------------------
WITH filtered_sessions AS (
    SELECT
        smba_id,
        application,
        TIMESTAMP_SECONDS(exposure_start_ts)       AS exposure_start_at,
        DATE(TIMESTAMP_SECONDS(exposure_start_ts)) AS exposure_date,
        exposure_end_ts - exposure_start_ts         AS duration_seconds
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE (exposure_end_ts - exposure_start_ts) > 60
),

first_per_service AS (
    SELECT
        smba_id,
        application AS service,
        exposure_start_at AS first_watch_at,
        exposure_date     AS first_watch_date
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY smba_id, application
                ORDER BY exposure_start_at ASC
            ) AS rn
        FROM filtered_sessions
        WHERE application IN (
            'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
            'Netflix', 'Hulu', 'Peacock', 'Paramount+'
        )
    )
    WHERE rn = 1
),

device_first_ever AS (
    SELECT
        smba_id,
        MIN(exposure_start_at) AS first_ever_activity_at
    FROM filtered_sessions
    GROUP BY smba_id
),

valid_first_watches AS (
    SELECT fps.*
    FROM first_per_service fps
    INNER JOIN device_first_ever dfe
        ON fps.smba_id = dfe.smba_id
    WHERE fps.first_watch_at > dfe.first_ever_activity_at
      AND fps.first_watch_date >= '2025-11-01'
),

-- For each valid first watch, look back 30 days and count distinct services
prior_service_counts AS (
    SELECT
        vfw.smba_id,
        vfw.service,
        COUNT(DISTINCT fs.application) AS distinct_services_30d
    FROM valid_first_watches vfw
    INNER JOIN filtered_sessions fs
        ON  vfw.smba_id = fs.smba_id
        AND fs.exposure_start_at < vfw.first_watch_at
        AND fs.exposure_start_at >= TIMESTAMP_SUB(vfw.first_watch_at, INTERVAL 30 DAY)
    GROUP BY vfw.smba_id, vfw.service
)

SELECT
    distinct_services_30d,
    COUNT(*) AS device_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM prior_service_counts
GROUP BY distinct_services_30d
ORDER BY distinct_services_30d
;


-- ---------------------------------------------------------------------------
-- 2. ACTIVITY LEVEL DISTRIBUTIONS (SESSIONS/DAY AND DURATION/DAY)
--    Measures how heavily a device uses streaming in the 30 days before
--    adopting a new service. High-activity users may gravitate toward
--    different titles (e.g., niche content) vs. casual users (who may
--    pick the biggest hit). Distributions help set bin boundaries and
--    detect outlier thresholds.
-- ---------------------------------------------------------------------------
WITH filtered_sessions AS (
    SELECT
        smba_id,
        application,
        TIMESTAMP_SECONDS(exposure_start_ts)       AS exposure_start_at,
        DATE(TIMESTAMP_SECONDS(exposure_start_ts)) AS exposure_date,
        exposure_end_ts - exposure_start_ts         AS duration_seconds
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE (exposure_end_ts - exposure_start_ts) > 60
),

first_per_service AS (
    SELECT
        smba_id,
        application AS service,
        exposure_start_at AS first_watch_at,
        exposure_date     AS first_watch_date
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY smba_id, application
                ORDER BY exposure_start_at ASC
            ) AS rn
        FROM filtered_sessions
        WHERE application IN (
            'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
            'Netflix', 'Hulu', 'Peacock', 'Paramount+'
        )
    )
    WHERE rn = 1
),

device_first_ever AS (
    SELECT
        smba_id,
        MIN(exposure_start_at) AS first_ever_activity_at
    FROM filtered_sessions
    GROUP BY smba_id
),

valid_first_watches AS (
    SELECT fps.*
    FROM first_per_service fps
    INNER JOIN device_first_ever dfe
        ON fps.smba_id = dfe.smba_id
    WHERE fps.first_watch_at > dfe.first_ever_activity_at
      AND fps.first_watch_date >= '2025-11-01'
),

-- Compute per-device activity metrics over the 30-day lookback window
prior_activity AS (
    SELECT
        vfw.smba_id,
        vfw.service,
        COUNT(*)                                                AS total_sessions_30d,
        COUNT(DISTINCT fs.exposure_date)                        AS active_days_30d,
        SUM(fs.duration_seconds)                                AS total_duration_30d,
        -- Normalize to daily rates for comparability across devices
        ROUND(COUNT(*) / 30.0, 2)                               AS sessions_per_day,
        ROUND(SUM(fs.duration_seconds) / 30.0, 1)               AS duration_per_day_seconds,
        ROUND(SUM(fs.duration_seconds) / 30.0 / 60.0, 1)        AS duration_per_day_minutes,
        COUNT(DISTINCT fs.application)                          AS distinct_services_30d,
        COUNT(DISTINCT CASE
            WHEN fs.application IN (
                'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
                'Netflix', 'Hulu', 'Peacock', 'Paramount+'
            ) THEN fs.application
        END)                                                    AS distinct_streaming_services_30d
    FROM valid_first_watches vfw
    INNER JOIN filtered_sessions fs
        ON  vfw.smba_id = fs.smba_id
        AND fs.exposure_start_at < vfw.first_watch_at
        AND fs.exposure_start_at >= TIMESTAMP_SUB(vfw.first_watch_at, INTERVAL 30 DAY)
    GROUP BY vfw.smba_id, vfw.service
)

-- Distribution summary using approximate percentiles for efficiency
SELECT
    'sessions_per_day'        AS metric,
    COUNT(*)                  AS n,
    ROUND(AVG(sessions_per_day), 2)          AS mean,
    ROUND(APPROX_QUANTILES(sessions_per_day, 100)[OFFSET(10)], 2) AS p10,
    ROUND(APPROX_QUANTILES(sessions_per_day, 100)[OFFSET(25)], 2) AS p25,
    ROUND(APPROX_QUANTILES(sessions_per_day, 100)[OFFSET(50)], 2) AS median,
    ROUND(APPROX_QUANTILES(sessions_per_day, 100)[OFFSET(75)], 2) AS p75,
    ROUND(APPROX_QUANTILES(sessions_per_day, 100)[OFFSET(90)], 2) AS p90,
    ROUND(APPROX_QUANTILES(sessions_per_day, 100)[OFFSET(99)], 2) AS p99,
    ROUND(MAX(sessions_per_day), 2)          AS max
FROM prior_activity

UNION ALL

SELECT
    'duration_per_day_minutes' AS metric,
    COUNT(*)                   AS n,
    ROUND(AVG(duration_per_day_minutes), 1),
    ROUND(APPROX_QUANTILES(duration_per_day_minutes, 100)[OFFSET(10)], 1),
    ROUND(APPROX_QUANTILES(duration_per_day_minutes, 100)[OFFSET(25)], 1),
    ROUND(APPROX_QUANTILES(duration_per_day_minutes, 100)[OFFSET(50)], 1),
    ROUND(APPROX_QUANTILES(duration_per_day_minutes, 100)[OFFSET(75)], 1),
    ROUND(APPROX_QUANTILES(duration_per_day_minutes, 100)[OFFSET(90)], 1),
    ROUND(APPROX_QUANTILES(duration_per_day_minutes, 100)[OFFSET(99)], 1),
    ROUND(MAX(duration_per_day_minutes), 1)
FROM prior_activity

UNION ALL

SELECT
    'active_days_30d'          AS metric,
    COUNT(*)                   AS n,
    ROUND(AVG(active_days_30d), 1),
    APPROX_QUANTILES(active_days_30d, 100)[OFFSET(10)],
    APPROX_QUANTILES(active_days_30d, 100)[OFFSET(25)],
    APPROX_QUANTILES(active_days_30d, 100)[OFFSET(50)],
    APPROX_QUANTILES(active_days_30d, 100)[OFFSET(75)],
    APPROX_QUANTILES(active_days_30d, 100)[OFFSET(90)],
    APPROX_QUANTILES(active_days_30d, 100)[OFFSET(99)],
    MAX(active_days_30d)
FROM prior_activity

UNION ALL

SELECT
    'total_sessions_30d'       AS metric,
    COUNT(*)                   AS n,
    ROUND(AVG(total_sessions_30d), 1),
    APPROX_QUANTILES(total_sessions_30d, 100)[OFFSET(10)],
    APPROX_QUANTILES(total_sessions_30d, 100)[OFFSET(25)],
    APPROX_QUANTILES(total_sessions_30d, 100)[OFFSET(50)],
    APPROX_QUANTILES(total_sessions_30d, 100)[OFFSET(75)],
    APPROX_QUANTILES(total_sessions_30d, 100)[OFFSET(90)],
    APPROX_QUANTILES(total_sessions_30d, 100)[OFFSET(99)],
    MAX(total_sessions_30d)
FROM prior_activity
;


-- ---------------------------------------------------------------------------
-- 3. CONTENT TYPE PREFERENCE DISTRIBUTIONS
--    What content types do devices watch in the 30 days before adopting a
--    new service? A device that mostly watches movies may choose a different
--    first title than one that watches series. This query computes each
--    device's dominant content type and the ratio of each type, then shows
--    the aggregate distribution.
-- ---------------------------------------------------------------------------
WITH filtered_sessions AS (
    SELECT
        smba_id,
        application,
        content_type,
        TIMESTAMP_SECONDS(exposure_start_ts)       AS exposure_start_at,
        DATE(TIMESTAMP_SECONDS(exposure_start_ts)) AS exposure_date,
        exposure_end_ts - exposure_start_ts         AS duration_seconds
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE (exposure_end_ts - exposure_start_ts) > 60
),

first_per_service AS (
    SELECT
        smba_id,
        application AS service,
        exposure_start_at AS first_watch_at,
        exposure_date     AS first_watch_date
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY smba_id, application
                ORDER BY exposure_start_at ASC
            ) AS rn
        FROM filtered_sessions
        WHERE application IN (
            'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
            'Netflix', 'Hulu', 'Peacock', 'Paramount+'
        )
    )
    WHERE rn = 1
),

device_first_ever AS (
    SELECT
        smba_id,
        MIN(exposure_start_at) AS first_ever_activity_at
    FROM filtered_sessions
    GROUP BY smba_id
),

valid_first_watches AS (
    SELECT fps.*
    FROM first_per_service fps
    INNER JOIN device_first_ever dfe
        ON fps.smba_id = dfe.smba_id
    WHERE fps.first_watch_at > dfe.first_ever_activity_at
      AND fps.first_watch_date >= '2025-11-01'
),

-- Count sessions by content type in the 30-day lookback per device
content_type_counts AS (
    SELECT
        vfw.smba_id,
        vfw.service,
        IFNULL(fs.content_type, '__NULL__') AS content_type,
        COUNT(*)                            AS session_count,
        SUM(fs.duration_seconds)            AS total_duration
    FROM valid_first_watches vfw
    INNER JOIN filtered_sessions fs
        ON  vfw.smba_id = fs.smba_id
        AND fs.exposure_start_at < vfw.first_watch_at
        AND fs.exposure_start_at >= TIMESTAMP_SUB(vfw.first_watch_at, INTERVAL 30 DAY)
    GROUP BY vfw.smba_id, vfw.service, fs.content_type
),

-- Determine each device's dominant content type (by session count)
device_dominant AS (
    SELECT
        smba_id,
        service,
        content_type AS dominant_content_type
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY smba_id, service
                ORDER BY session_count DESC
            ) AS rn
        FROM content_type_counts
    )
    WHERE rn = 1
),

-- Compute content type ratios per device (percentage of sessions for each type)
device_totals AS (
    SELECT
        smba_id,
        service,
        SUM(session_count) AS total_sessions
    FROM content_type_counts
    GROUP BY smba_id, service
),

device_ratios AS (
    SELECT
        ctc.smba_id,
        ctc.service,
        ctc.content_type,
        ROUND(ctc.session_count / dt.total_sessions, 3) AS content_ratio
    FROM content_type_counts ctc
    INNER JOIN device_totals dt
        ON  ctc.smba_id = dt.smba_id
        AND ctc.service  = dt.service
)

-- 3a. Dominant content type distribution: what type do most devices prefer?
SELECT
    'dominant_content_type' AS analysis,
    dominant_content_type   AS value,
    COUNT(*)                AS device_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_devices
FROM device_dominant
GROUP BY dominant_content_type
ORDER BY device_count DESC
;

-- 3b. Content ratio percentile distributions per content type.
--     Shows how concentrated or spread out content preferences are.
--     E.g., if the median "Movie" ratio is 0.6, most devices watch 60% movies.
WITH filtered_sessions AS (
    SELECT
        smba_id,
        application,
        content_type,
        TIMESTAMP_SECONDS(exposure_start_ts)       AS exposure_start_at,
        DATE(TIMESTAMP_SECONDS(exposure_start_ts)) AS exposure_date,
        exposure_end_ts - exposure_start_ts         AS duration_seconds
    FROM `antenna-reporting.samba_tv.streaming_viewership_2_1_inc`
    WHERE (exposure_end_ts - exposure_start_ts) > 60
),

first_per_service AS (
    SELECT
        smba_id,
        application AS service,
        exposure_start_at AS first_watch_at,
        exposure_date     AS first_watch_date
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY smba_id, application
                ORDER BY exposure_start_at ASC
            ) AS rn
        FROM filtered_sessions
        WHERE application IN (
            'Amazon Prime Video', 'Apple TV+', 'Disney+', 'Max',
            'Netflix', 'Hulu', 'Peacock', 'Paramount+'
        )
    )
    WHERE rn = 1
),

device_first_ever AS (
    SELECT
        smba_id,
        MIN(exposure_start_at) AS first_ever_activity_at
    FROM filtered_sessions
    GROUP BY smba_id
),

valid_first_watches AS (
    SELECT fps.*
    FROM first_per_service fps
    INNER JOIN device_first_ever dfe
        ON fps.smba_id = dfe.smba_id
    WHERE fps.first_watch_at > dfe.first_ever_activity_at
      AND fps.first_watch_date >= '2025-11-01'
),

content_type_counts AS (
    SELECT
        vfw.smba_id,
        vfw.service,
        IFNULL(fs.content_type, '__NULL__') AS content_type,
        COUNT(*)                            AS session_count
    FROM valid_first_watches vfw
    INNER JOIN filtered_sessions fs
        ON  vfw.smba_id = fs.smba_id
        AND fs.exposure_start_at < vfw.first_watch_at
        AND fs.exposure_start_at >= TIMESTAMP_SUB(vfw.first_watch_at, INTERVAL 30 DAY)
    GROUP BY vfw.smba_id, vfw.service, fs.content_type
),

device_totals AS (
    SELECT
        smba_id,
        service,
        SUM(session_count) AS total_sessions
    FROM content_type_counts
    GROUP BY smba_id, service
),

device_ratios AS (
    SELECT
        ctc.content_type,
        ROUND(ctc.session_count / dt.total_sessions, 3) AS content_ratio
    FROM content_type_counts ctc
    INNER JOIN device_totals dt
        ON  ctc.smba_id = dt.smba_id
        AND ctc.service  = dt.service
)

SELECT
    content_type,
    COUNT(*)                                                            AS n,
    ROUND(AVG(content_ratio), 3)                                        AS mean_ratio,
    ROUND(APPROX_QUANTILES(content_ratio, 100)[OFFSET(25)], 3)         AS p25,
    ROUND(APPROX_QUANTILES(content_ratio, 100)[OFFSET(50)], 3)         AS median_ratio,
    ROUND(APPROX_QUANTILES(content_ratio, 100)[OFFSET(75)], 3)         AS p75,
    ROUND(APPROX_QUANTILES(content_ratio, 100)[OFFSET(90)], 3)         AS p90
FROM device_ratios
GROUP BY content_type
HAVING COUNT(*) >= 100    -- only show types with meaningful sample size
ORDER BY n DESC
;
