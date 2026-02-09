---
name: logql
description: LogQL (Loki Query Language) reference — stream selectors, line/label filters, parsers, metric queries, and patterns. Use when writing or debugging LogQL queries for Grafana Loki log analysis, dashboards, or alerting.
---

# LogQL

## Query Structure

```
{stream selector} | pipeline stage | pipeline stage | ...
```

Two query types:
- **Log queries**: Return log lines (with optional filtering/parsing)
- **Metric queries**: Return numeric values computed from log lines

## Stream Selector

Mandatory. Selects log streams by label:

```logql
{namespace="production", app="api-server"}
{job=~"ingress-.+", cluster!="staging"}
{container!~"sidecar|init-.+"}
```

| Operator | Meaning |
|----------|---------|
| `=` | Exact match |
| `!=` | Not equal |
| `=~` | Regex match (fully anchored, RE2 syntax) |
| `!~` | Regex not match |

**Performance**: Stream selectors are the primary index lookup. Narrow selectors = faster queries. Always specify at least one exact-match label.

## Log Pipeline

### Line Filter Expressions

Distributed grep — fastest filter stage:

```logql
{app="api"} |= "error"                 # Contains string
{app="api"} != "healthcheck"            # Does not contain
{app="api"} |~ "status=(4|5)\\d{2}"    # Regex match
{app="api"} !~ "GET /health"           # Regex not match
```

**Chain filters** — all must match (AND):

```logql
{app="api"} |= "error" != "timeout" |~ "user_id=\\d+"
```

**Performance tip**: Place line filters before parsers. They're the cheapest filter operation.

**Case-insensitive**: Prefix regex with `(?i)`:

```logql
{app="api"} |~ "(?i)error"
```

**Backtick quoting** (avoids double-escaping):

```logql
{app="api"} |~ `status=\d{3}`
```

### Decolorize

Strip ANSI color codes:

```logql
{app="api"} | decolorize
```

### Parser Expressions

Extract labels from log content for filtering and aggregation.

#### JSON

```logql
# Extract all fields
{app="api"} | json

# Extract specific fields
{app="api"} | json method="request.method", status="response.status"

# Array access
{app="api"} | json first_ip="ips[0]"
```

Nested keys are flattened: `request.headers.User_Agent` → `request_headers_User_Agent`.

#### logfmt

```logql
# Parse all key=value pairs
{app="api"} | logfmt

# Extract specific keys
{app="api"} | logfmt --strict duration, status

# --strict: drop lines that fail to parse (instead of adding __error__)
```

#### pattern

Fast template-based extraction:

```logql
# Apache common log format
{job="nginx"} | pattern `<ip> - <user> [<_>] "<method> <path> <_>" <status> <size>`

# Capture between static markers
{app="api"} | pattern `level=<level> msg="<msg>"`
```

`<_>` discards a field. `<name>` captures into label `name`. Static text between captures must match literally.

#### regexp

Full regex extraction (slower than pattern):

```logql
{app="api"} | regexp `(?P<method>\w+)\s+(?P<path>\S+)\s+HTTP/\d\.\d"\s+(?P<status>\d{3})`
```

Named capture groups become labels.

#### unpack

For logs packed by Promtail's pack stage (JSON with `_entry` key):

```logql
{app="api"} | unpack
```

### Label Filter Expressions

Filter on extracted or original labels. Supports typed comparisons:

```logql
# String comparison
{app="api"} | json | status != "200"
{app="api"} | json | method =~ "GET|POST"

# Numeric comparison
{app="api"} | json | status >= 400
{app="api"} | logfmt | bytes > 1024

# Duration comparison
{app="api"} | logfmt | duration > 10s
{app="api"} | logfmt | duration >= 1m30s

# Byte comparison
{app="api"} | logfmt | size > 5MB

# Boolean logic
{app="api"} | json | status >= 500 or duration > 5s
{app="api"} | json | status >= 400 and status < 500 , method = "POST"
```

`and`, `,`, ` ` (space), and `|` between label filters all mean AND. Use `or` for OR. Parentheses control precedence (left-to-right by default).

### Format Expressions

#### line_format

Rewrite the log line using Go template syntax:

```logql
{app="api"} | json | line_format "{{.method}} {{.path}} → {{.status}} ({{.duration}})"

# With template functions
{app="api"} | json | line_format `{{ .timestamp | toDateInZone "2006-01-02" "UTC" }}: {{ .message }}`
```

Available functions: `ToLower`, `ToUpper`, `Replace`, `Trim`, `TrimLeft`, `TrimRight`, `TrimPrefix`, `TrimSuffix`, `TrimSpace`, `regexReplaceAll`, `regexReplaceAllLiteral`, `toDateInZone`, `unixEpoch`.

#### label_format

Rename or modify labels:

```logql
{app="api"} | json | label_format dst=src
{app="api"} | logfmt | label_format duration_ms="{{divide .duration_seconds 0.001}}"
```

### Drop / Keep Labels

Control which labels appear in results:

```logql
{app="api"} | json | keep method, status, duration
{app="api"} | json | drop __error__, __error_details__
```

## Metric Queries

### Log Range Aggregations

Operate on log lines (no unwrap needed):

| Function | Description |
|----------|-------------|
| `rate({...}[d])` | Entries per second |
| `count_over_time({...}[d])` | Total entries in window |
| `bytes_rate({...}[d])` | Bytes per second |
| `bytes_over_time({...}[d])` | Total bytes in window |
| `absent_over_time({...}[d])` | 1 if no entries in window |

```logql
# Error rate per second
rate({app="api"} |= "error" [5m])

# Error count by namespace
sum by (namespace) (count_over_time({app="api"} |= "error" [1h]))

# Log volume in bytes
sum by (app) (bytes_rate({namespace="production"}[5m]))
```

### Unwrapped Range Aggregations

Extract a numeric label value and aggregate it:

```logql
# Average request duration from logfmt logs
avg_over_time({app="api"} | logfmt | unwrap duration [5m])

# P99 response size
quantile_over_time(0.99, {app="api"} | json | unwrap response_size [5m])

# Max latency per endpoint
max_over_time({app="api"} | logfmt | unwrap latency | latency > 0 [5m]) by (endpoint)
```

| Function | Description |
|----------|-------------|
| `rate(... \| unwrap x [d])` | Per-second rate of sum of values |
| `sum_over_time(... \| unwrap x [d])` | Sum of values |
| `avg_over_time(... \| unwrap x [d])` | Average |
| `min_over_time(... \| unwrap x [d])` | Minimum |
| `max_over_time(... \| unwrap x [d])` | Maximum |
| `first_over_time(... \| unwrap x [d])` | First value |
| `last_over_time(... \| unwrap x [d])` | Last value |
| `stddev_over_time(... \| unwrap x [d])` | Standard deviation |
| `quantile_over_time(φ, ... \| unwrap x [d])` | φ-quantile |
| `rate_counter(... \| unwrap x [d])` | Per-second rate treating values as counter |

**Duration conversion**: Use `| unwrap duration_seconds(label)` or `| unwrap bytes(label)` for automatic unit conversion.

### Aggregation Operators

Same as PromQL:

```logql
sum, avg, min, max, count, topk, bottomk, stddev, stdvar, sort, sort_desc
```

With grouping:

```logql
sum by (status) (rate({app="api"} | json [5m]))
topk(5, sum by (path) (rate({app="api"} | json [5m])))
```

### Offset Modifier

```logql
# Compare current error rate to 1 day ago
count_over_time({app="api"} |= "error" [1h])
/
count_over_time({app="api"} |= "error" [1h] offset 1d)
```

## Practical Patterns

### K8s Pod Error Analysis

```logql
# Errors in a specific namespace, parsed and filtered
{namespace="ml-training"} |= "error" | json | level="error"
  | line_format "{{.pod}} {{.message}}"

# OOMKilled detection
{namespace="ml-training"} |= "OOMKilled" or |= "Out of memory"

# GPU errors in training pods
{namespace="ml-training", container="trainer"} |~ "CUDA|cuda|GPU|nccl"
  | json | level =~ "error|fatal"
```

### Log-Based Metrics for Dashboards

```logql
# Request rate by status class
sum by (status_class) (
  rate({app="api"} | json | label_format status_class="{{regexReplaceAll `(\\d)\\d{2}` .status `${1}xx`}}" [5m])
)

# P95 latency from access logs
quantile_over_time(0.95,
  {app="api"} | logfmt | unwrap duration | duration > 0 [5m]
) by (method)
```

### Log-Based Alerting

```yaml
# Loki ruler config
groups:
  - name: app_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum by (namespace, app) (rate({namespace="production"} |= "level=error" [5m]))
          > 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in {{ $labels.app }}"

      - alert: GPUErrorDetected
        expr: |
          count_over_time({namespace="ml-training"} |~ "CUDA error|GPU fault" [10m])
          > 0
        labels:
          severity: warning
```

### IP Address Matching

```logql
{app="nginx"} | logfmt | addr = ip("10.0.0.0/8")          # Match CIDR range
{app="nginx"} | logfmt | addr != ip("192.168.0.0/16")      # Exclude range
```

## Query Optimization

| Technique | Impact | How |
|-----------|--------|-----|
| Narrow stream selector | High | Add more exact-match labels |
| Line filter before parser | High | `\|= "error" \| json` not `\| json \|= "error"` |
| Extract only needed fields | Medium | `\| json status, duration` not `\| json` |
| Shorter time ranges | Medium | Query only what you need |
| Use `pattern` over `regexp` | Medium | Faster parsing |
| Recording rules | High | Pre-compute expensive metric queries |
| Use `--strict` with logfmt | Low | Drops unparseable lines early |

## LogQL vs PromQL

| Feature | PromQL | LogQL |
|---------|--------|-------|
| Data source | Metrics (time series) | Logs (text streams) |
| Selectors | `metric{label="value"}` | `{label="value"}` (no metric name) |
| Rate input | Counter time series | Log stream |
| Parsing | N/A (structured data) | json, logfmt, pattern, regexp |
| Line filtering | N/A | `\|=`, `!=`, `\|~`, `!~` |
| Unwrap | N/A | Extract numeric values from logs |
| Aggregations | Same operators | Same operators |
| Subqueries | `func(v[range:step])` | Same syntax |

## Cross-References

- [promql](../promql/) — PromQL for Prometheus metrics (LogQL is based on PromQL)
