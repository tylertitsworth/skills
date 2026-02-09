---
name: promql
description: PromQL (Prometheus Query Language) reference — selectors, operators, functions, aggregations, and query patterns. Use when writing or debugging PromQL queries for Prometheus metrics, Grafana dashboards, or alerting rules.
---

# PromQL

## Data Types

| Type | Description | Example |
|------|-------------|---------|
| Instant vector | Set of time series, each with one sample at query time | `http_requests_total{job="api"}` |
| Range vector | Set of time series, each with a range of samples | `http_requests_total{job="api"}[5m]` |
| Scalar | Single numeric value | `42`, `3.14` |
| String | String value (limited use) | `"hello"` |

## Selectors

### Instant Vector Selector

```promql
metric_name{label1="value1", label2=~"regex.*"}
```

| Operator | Meaning |
|----------|---------|
| `=` | Exact match |
| `!=` | Not equal |
| `=~` | Regex match (fully anchored) |
| `!~` | Regex not match (fully anchored) |

Regex uses RE2 syntax. `=~".*"` matches any value (including empty). `=~".+"` matches non-empty.

### Range Vector Selector

Append `[duration]` to get samples over a time window:

```promql
http_requests_total{job="api"}[5m]
```

Duration units: `ms`, `s`, `m`, `h`, `d`, `w`, `y`.

### Offset and @ Modifiers

```promql
# Value from 1 hour ago
http_requests_total offset 1h

# Value at specific Unix timestamp
http_requests_total @ 1609459200

# Combine: range ending 1h ago
rate(http_requests_total[5m] offset 1h)

# @ with range
rate(http_requests_total[5m] @ 1609459200)
```

## Operators

### Arithmetic

`+`, `-`, `*`, `/`, `%`, `^` (power)

Between two instant vectors, matching is by label. Between vector and scalar, applied to every sample.

### Comparison

`==`, `!=`, `>`, `<`, `>=`, `<=`

By default, filter (drop non-matching series). Add `bool` to return 0/1:

```promql
http_requests_total > 100          # Filters
http_requests_total > bool 100     # Returns 0 or 1
```

### Logical/Set

```promql
vector1 and vector2     # Intersection (keep left where right exists)
vector1 or vector2      # Union (left + unmatched right)
vector1 unless vector2  # Complement (left where right doesn't exist)
```

### Vector Matching

```promql
# One-to-one (default): match on all labels
metric_a / metric_b

# Match on specific labels only
metric_a / on(job, instance) metric_b

# Match on all labels except
metric_a / ignoring(status_code) metric_b

# Many-to-one / one-to-many
metric_a / on(job) group_left(extra_label) metric_b
metric_a / on(job) group_right(extra_label) metric_b
```

`group_left` = many-to-one (left has more series). `group_right` = one-to-many. Extra labels from the "one" side can be copied into the result.

### Merging Metrics with Different Label Names

A common problem: two metrics describe the same entity but use different label names (e.g., `hostname` vs `node_name` vs `node` vs `instance`). PromQL has no direct "join on different label names" — you must relabel first.

**Pattern 1: `label_replace` to create a shared label**

```promql
# GPU metrics use "Hostname", node metrics use "node"
# Create a common "node" label on GPU metrics, then join
label_replace(DCGM_FI_DEV_GPU_UTIL, "node", "$1", "Hostname", "(.*)")
  * on(node) group_left()
kube_node_info
```

**Pattern 2: `label_join` to combine labels**

```promql
# Create a composite label from multiple source labels
label_join(metric, "combined", "-", "namespace", "pod")
```

**Pattern 3: Use an info metric as a lookup table**

Many exporters provide "info" metrics (value always 1) with multiple labels. Use `group_left` to enrich:

```promql
# kube_pod_info has labels: pod, node, namespace, host_ip, pod_ip
# DCGM metrics have: pod, namespace, gpu
# Join to get node name on GPU metrics
DCGM_FI_DEV_GPU_UTIL
  * on(namespace, pod) group_left(node)
kube_pod_info
```

**Pattern 4: Chained `label_replace` for complex mappings**

```promql
# node_exporter uses "instance" (ip:port), kube metrics use "node" (hostname)
# Step 1: Extract hostname from instance
label_replace(
  node_memory_MemAvailable_bytes,
  "node", "$1", "instance", "(.*):.*"
)
# Step 2: Join with kube metrics
* on(node) group_left()
kube_node_status_capacity{resource="memory"}
```

**Common label name mismatches in ML/K8s environments**:

| Source | Label | Contains |
|--------|-------|----------|
| DCGM exporter | `Hostname` | Node hostname |
| node_exporter | `instance` | `hostname:port` |
| kube-state-metrics | `node` | Node name |
| kube-state-metrics | `pod` | Pod name |
| cAdvisor | `container`, `pod` | Container/pod name |
| NVIDIA GPU Operator | `gpu`, `UUID` | GPU index/UUID |

## Aggregation Operators

```promql
<aggr>([parameter,] vector) [by|without (labels)]
```

| Operator | Description |
|----------|-------------|
| `sum` | Sum over dimensions |
| `avg` | Average |
| `min` / `max` | Minimum / maximum |
| `count` | Count of series |
| `count_values("label", vector)` | Count series with each distinct value, stored in `label` |
| `stddev` / `stdvar` | Standard deviation / variance |
| `topk(k, vector)` | Largest k elements |
| `bottomk(k, vector)` | Smallest k elements |
| `quantile(φ, vector)` | φ-quantile across series |
| `group` | Returns 1 for each group (useful for existence checks) |

```promql
# Sum by job
sum by (job) (rate(http_requests_total[5m]))

# Sum dropping instance label
sum without (instance) (rate(http_requests_total[5m]))

# Top 5 by request rate
topk(5, rate(http_requests_total[5m]))
```

## Functions

### Rate / Increase / Delta

| Function | Input | Output | Use With |
|----------|-------|--------|----------|
| `rate(v[d])` | Counter range | Per-second rate | Counters (e.g., `_total`) |
| `irate(v[d])` | Counter range | Instant rate (last two points) | High-resolution, volatile |
| `increase(v[d])` | Counter range | Total increase over range | Counters |
| `delta(v[d])` | Gauge range | Difference first→last | Gauges |
| `idelta(v[d])` | Gauge range | Difference last two points | Gauges |
| `deriv(v[d])` | Gauge range | Per-second derivative (linear regression) | Gauges |

**`rate` vs `irate`**: `rate` averages over the entire range window — smoother, better for alerting. `irate` uses only the last two samples — more responsive but noisier. Almost always prefer `rate`.

**Counter resets**: `rate` and `increase` handle counter resets automatically (monotonic counter that drops to 0 on restart).

### Aggregation Over Time

Apply to range vectors from a single series:

| Function | Effect |
|----------|--------|
| `avg_over_time(v[d])` | Average over window |
| `min_over_time(v[d])` | Minimum |
| `max_over_time(v[d])` | Maximum |
| `sum_over_time(v[d])` | Sum |
| `count_over_time(v[d])` | Count of samples |
| `quantile_over_time(φ, v[d])` | φ-quantile |
| `stddev_over_time(v[d])` | Standard deviation |
| `stdvar_over_time(v[d])` | Variance |
| `last_over_time(v[d])` | Most recent sample |
| `present_over_time(v[d])` | 1 for any series that has data |

### Histogram Functions

```promql
# Classic histogram: 90th percentile of request duration
histogram_quantile(0.9, rate(http_request_duration_seconds_bucket[5m]))

# Aggregate across instances, preserving le bucket label
histogram_quantile(0.9,
  sum by (le, job) (rate(http_request_duration_seconds_bucket[5m]))
)

# Native histogram: no _bucket suffix needed
histogram_quantile(0.9, rate(http_request_duration_seconds[5m]))

# Fraction of requests under 200ms
histogram_fraction(0, 0.2, rate(http_request_duration_seconds[5m]))

# Average from histogram
histogram_avg(rate(http_request_duration_seconds[5m]))
```

**Critical**: When aggregating classic histograms with `sum`, always preserve the `le` label — it identifies bucket boundaries.

### Prediction / Smoothing

```promql
# Linear prediction: predict value 4h from now based on last 1d
predict_linear(node_filesystem_avail_bytes[1d], 4*3600)

# Exponential smoothing
double_exponential_smoothing(v[d], smoothing_factor, trend_factor)
```

### Existence / Absence

```promql
# Alert when metric disappears
absent(up{job="api"})                    # Returns {job="api"} with value 1 if no data
absent_over_time(up{job="api"}[5m])      # Same but over a window
```

### Math Functions

`abs`, `ceil`, `floor`, `round`, `exp`, `ln`, `log2`, `log10`, `sqrt`, `sgn`, `clamp(v, min, max)`, `clamp_min`, `clamp_max`

### Time Functions

`time()`, `timestamp(v)`, `day_of_week()`, `day_of_month()`, `day_of_year()`, `days_in_month()`, `hour()`, `minute()`, `month()`, `year()`

### Label Functions

```promql
# Manipulate labels
label_replace(v, "dst_label", "$1", "src_label", "regex")
label_join(v, "dst_label", "separator", "src1", "src2")
```

### Sorting

```promql
sort(v)       # Ascending by sample value
sort_desc(v)  # Descending
sort_by_label(v, "label1", "label2")       # Ascending by label
sort_by_label_desc(v, "label1", "label2")  # Descending by label
```

## Subqueries

Apply instant-query functions over a range:

```promql
# Max of 5m rate, evaluated every 1m over last 1h
max_over_time(rate(http_requests_total[5m])[1h:1m])
#                                          [range:resolution]
```

If resolution is omitted, it uses the global evaluation interval.

## Common Patterns

### Error Rate

```promql
# HTTP error rate (%)
sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
/
sum by (job) (rate(http_requests_total[5m]))
* 100
```

### Saturation

```promql
# CPU saturation
1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))

# Memory saturation
1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)
```

### Latency SLO

```promql
# Fraction of requests under 300ms (Apdex-style)
sum(rate(http_request_duration_seconds_bucket{le="0.3"}[5m]))
/
sum(rate(http_request_duration_seconds_count[5m]))
```

### GPU Metrics (DCGM)

```promql
# GPU utilization per pod
avg by (pod, gpu) (DCGM_FI_DEV_GPU_UTIL)

# GPU memory used
DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100

# GPU temperature alert threshold
DCGM_FI_DEV_GPU_TEMP > 85

# Tensor core utilization
avg by (pod) (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)
```

### Absent / Dead Series Detection

```promql
# Fire alert if no GPU metrics for 5 minutes
absent(DCGM_FI_DEV_GPU_UTIL{job="dcgm-exporter"})

# Count of healthy targets
count(up{job="api"} == 1)
```

## Recording Rules

Pre-compute expensive queries as new time series:

```yaml
groups:
  - name: api_rules
    interval: 30s
    rules:
      - record: job:http_requests:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      - record: job:http_request_duration:p99
        expr: histogram_quantile(0.99, sum by (le, job) (rate(http_request_duration_seconds_bucket[5m])))
```

**Naming convention**: `level:metric:operations` — e.g., `job:http_requests:rate5m`.

## Alerting Rules

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          /
          sum by (job) (rate(http_requests_total[5m]))
          > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.job }}"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: GPUTemperatureHigh
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 10m
        labels:
          severity: warning
```

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| `rate(gauge_metric[5m])` | rate is for counters only | Use `deriv()` or `delta()` |
| `sum(http_requests_total{status=~"5.."})` | Summing raw counter | Wrap in `rate()` first |
| `histogram_quantile(0.9, sum(rate(..._bucket[5m])))` | Dropped `le` label | `sum by (le) (...)` |
| `rate(metric[1s])` | Range smaller than scrape interval | Use range ≥ 4x scrape interval |
| `irate()` in alerting rules | Too volatile | Use `rate()` |
| Very high cardinality in `by()` | Query timeout | Reduce label dimensions, use recording rules |

## Cross-References

- [logql](../logql/) — LogQL for Loki log queries (PromQL-inspired syntax)
