# GreenBoost Monitoring Integration Guide

## Overview

This guide explains how to integrate GreenBoost's tiered memory metrics with Prometheus on a Kubernetes cluster running GreenBoost with DRA support.

## Prerequisites

- Kubernetes 1.32+ with DRA enabled
- GreenBoost kernel module loaded (`greenboost.ko`)
- GreenBoost DRA driver installed (greenboost-kubelet-plugin DaemonSet)
- Prometheus Operator deployed

## Component Metrics

### GreenBoost Metrics Exporter

The GreenBoost metrics exporter exposes metrics at `:8080/metrics`:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `greenboost_t1_used_bytes` | Gauge | Physical VRAM used (per GPU) |
| `greenboost_t1_total_bytes` | Gauge | Physical VRAM total |
| `greenboost_t2_used_bytes` | Gauge | System DDR pool used |
| `greenboost_t2_total_bytes` | Gauge | DDR pool total |
| `greenboost_t3_used_bytes` | Gauge | NVMe/Lustre swap used |
| `greenboost_t3_total_bytes` | Gauge | T3 total capacity |
| `greenboost_allocations_total` | Counter | Total allocations |
| `greenboost_evictions_total` | Counter | T3 spilling event count |
| `greenboost_watchdog_pressure` | Gauge | Watchdog pressure (0-100) |
| `greenboost_nvlink_ready` | Gauge | NVLink fabric health (0/1) |

### Example Metrics Output

```
# HELP greenboost_t1_used_bytes Physical VRAM in use (T1)
# TYPE greenboost_t1_used_bytes gauge
greenboost_t1_used_bytes{gpu="0"} 10737418240
greenboost_t1_used_bytes{gpu="1"} 10737418240
# TYPE greenboost_t1_total_bytes gauge
greenboost_t1_total_bytes 32984999936

# HELP greenboost_t2_used_bytes System DDR pool used (T2)
# TYPE greenboost_t2_used_bytes gauge
greenboost_t2_used_bytes 329853488128

# HELP greenboost_t2_total_bytes DDR pool total capacity
# TYPE greenboost_t2_total_bytes gauge
greenboost_t2_total_bytes 329853488128

# HELP greenboost_watchdog_pressure Watchdog pressure (0=healthy, 100=impending OOM)
# TYPE greenboost_watchdog_pressure gauge
greenboost_watchdog_pressure 0

# HELP greenboost_nvlink_ready NVLink fabric health (1=ready, 0=not ready)
# TYPE greenboost_nvlink_ready gauge
greenboost_nvlink_ready 1
```

## ServiceMonitor Configuration

Create a ServiceMonitor to scrape GreenBoost metrics:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: greenboost-metrics
  namespace: monitoring
  labels:
    app: greenboost
spec:
  selector:
    matchLabels:
      app: greenboost-metrics-exporter
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
  namespaceSelector:
    matchNames:
    - greenboost-system
```

## Grafana Dashboard Integration

### Dashboard JSON

Import the following dashboard to visualize GreenBoost metrics:

```json
{
  "dashboard": {
    "title": "GreenBoost Cluster Memory",
    "panels": [
      {
        "title": "Tiered Memory Usage",
        "targets": [
          {
            "expr": "sum(greenboost_t1_used_bytes) / 1024^3",
            "legendFormat": "T1 VRAM (GB)"
          },
          {
            "expr": "greenboost_t2_used_bytes / 1024^3",
            "legendFormat": "T2 DDR (GB)"
          },
          {
            "expr": "greenboost_t3_used_bytes / 1024^3",
            "legendFormat": "T3 Swap (GB)"
          }
        ]
      },
      {
        "title": "NVLink Fabric Health",
        "targets": [
          {
            "expr": "avg(greenboost_nvlink_ready) * 100",
            "legendFormat": "NVLink Ready %"
          }
        ]
      },
      {
        "title": "Watchdog Pressure",
        "targets": [
          {
            "expr": "avg(greenboost_watchdog_pressure)",
            "legendFormat": "Pressure Level"
          },
          {
            "expr": "avg(greenboost_watchdog_pressure) > 90",
            "legendFormat": "Critical Alert"
          }
        ]
      }
    ]
  }
}
```

## Alerting Rules

Create a PrometheusRule for critical alerts:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: greenboost-alerts
  namespace: monitoring
spec:
  groups:
  - name: greenboost.rules
    rules:
    - alert: GreenBoostNVLinkDown
      expr: avg(greenboost_nvlink_ready) < 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "NVLink fabric not ready on cluster"
        description: "NVLink fabric state indicates {{ $value }} nodes without healthy NVLink"

    - alert: GreenBoostHighWatchdogPressure
      expr: avg(greenboost_watchdog_pressure) > 80
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "GreenBoost watchdog pressure high"
        description: "Watchdog pressure is {{ $value }} (approaching OOM)"

    - alert: GreenBoostT3Spiking
      expr: rate(greenboost_evictions_total[5m]) > 10
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "T3 evictions spiking"
        description: "T3 spilling rate is {{ $value }} per second"
```

## NVIDIA GPU Dashboard Integration

Add GreenBoost metrics to the existing NVIDIA DCGM Exporter dashboard:

1. In the GPU Memory panel, add:
   - GreenBoost T1 usage overlay
   - T2 DDR pool as secondary series

2. Create new panels for:
   - GreenBoost T3 swap usage per node
   - Watchdog pressure history
   - NVLink fabric health across cluster

## Query Examples

### Cluster-Wide Virtual VRAM

```promql
# Total virtual VRAM across all nodes
sum(greenboost_t1_total_bytes + greenboost_t2_total_bytes) / 1024^3

# Cluster aggregate (T1 + T2 per node × 200 nodes)
sum(greenboost_t1_total_bytes) + sum(greenboost_t2_total_bytes)
```

### Per-Node Efficiency

```promql
# T1 usage percentage
sum(greenboost_t1_used_bytes) by (instance) / sum(greenboost_t1_total_bytes) by (instance) * 100

# T2 usage percentage
greenboost_t2_used_bytes / greenboost_t2_total_bytes * 100
```

### ComputeDomain Detection

```promql
# Detect if pods are running in ComputeDomain (from compute_domain_active)
avg(compute_domain_active) by (instance)
```

## Scaling to 200 Nodes

### Best Practices

1. **Metric Retention:**
   - Keep detailed metrics for 7 days
   - Retain aggregated data for 30 days

2. **Scraping Configuration:**
   - Scrape interval: 15s
   - Timeout: 10s
   - Parallelism (if using Thanos): 4 shards

3. **Query Performance:**
   - Use recording rules for frequently queried aggregations
   - Enable query caching
   - Consider index for Node label queries

## Installation Checklist

1. Deploy GreenBoost DRA driver:
   ```bash
   helm install greenboost-dra deployments/helm/greenboost-dra-driver
   ```

2. Deploy metrics exporter:
   ```bash
   kubectl apply -f deployments/helm/greenboost-dra-driver/exporter/
   ```

3. Create ServiceMonitor:
   ```bash
   kubectl apply -f k8s-deployment/servicemonitor.yaml
   ```

4. Import Grafana dashboard:
   - Go to Grafana → Dashboards → Import
   - Paste dashboard JSON
   - Select Prometheus data source

5. Verify metrics:
   ```bash
   kubectl port-forward -n greenboost-system svc/greenboost-metrics 8080:8080
   curl http://localhost:8080/metrics
   ```

## Troubleshooting

### Metrics Not Appearing

1. Check exporter pod status:
   ```bash
   kubectl get pods -n greenboost-system -l app=greenboost-metrics-exporter
   ```

2. Check ServiceMonitor is targeting correctly:
   ```bash
   kubectl get servicemonitor greenboost-metrics -n monitoring -o yaml
   ```

3. Verify Prometheus service discovery:
   ```bash
   kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
   # Go to http://localhost:9090/targets
   ```

### High Latency Queries

- Increase scrape timeout
- Enable query cache in Prometheus
- Use recording rules for complex aggregations

## Documentation References

- [K8s-dra-driver-gpu Metrics](https://github.com/NVIDIA/k8s-dra-driver-gpu/tree/main/cmd/compute-domain-controller)
- [Prometheus Operator Best Practices](https://prometheus-operator.dev/docs/)
- [GreenBoost Architecture](architecture.md)