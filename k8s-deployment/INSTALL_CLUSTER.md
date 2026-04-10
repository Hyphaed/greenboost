# GreenBoost Cluster Installation Guide

## Overview

This guide provides step-by-step instructions for deploying GreenBoost on a 200-node V100 NVLink cluster with NVIDIA's k8s-dra-driver-gpu integration.

## Cluster Specifications

**Per Node:**
- 8× NVIDIA V100 SXM2 32 GB (256 GB HBM2 total)
- 384 GB DDR4 ECC RAM
- 2× 1 TB Enterprise U.2 NVMe (RAID1)
- 200 Gb/s Mellanox ConnectX-6 InfiniBand

**Cluster Aggregate:**
- 200 nodes, 1,600 GPUs
- 51.2 TB physical GPU memory
- 76.8 TB system RAM
- **112 TB virtual VRAM** (2.2× physical)
- 200 PFLOPS FP16 AI compute
- 200 Gb/s HDR InfiniBand fabric

## Prerequisites

### System Requirements

1. **Kubernetes:**
   - Version 1.32 or higher
   - DRA (Dynamic Resource Allocation) enabled

2. **NVIDIA Software:**
   - NVIDIA GPU Operator installed
   - NVIDIA driver 580.x or later
   - CUDA 13.x

3. **GreenBoost:**
   - GreenBoost kernel module source code
   - Go 1.20+ for building kubelet plugin

4. **Storage:**
   - Lustre parallel filesystem installed
   - At least 2 PB total capacity for T3 tier

### Network Configuration

- HDR InfiniBand fabric operational
- Node IP assignment complete
- Firewall rules for internal communication

## Installation Steps

### Step 1: Install NVIDIA GPU Operator

```bash
# NVIDIA GPU Operator installation
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator with DRA enabled
helm install --wait --generate-name \
  nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set migStrategy=single \
  --set dcgmExporter.enabled=true
```

### Step 2: Install NVIDIA k8s-dra-driver-gpu

```bash
# Clone k8s-dra-driver-gpu repository (if needed)
git clone https://github.com/NVIDIA/k8s-dra-driver-gpu.git \
  ~/Dev/greenboost_sources/k8s-dra-driver-gpu
cd ~/Dev/greenboost_sources/k8s-dra-driver-gpu

# Install via Helm
helm install --wait --generate-name \
  deployments/helm/nvidia-dra-driver-gpu \
  --namespace nvidia-dra-system \
  --create-namespace \
  --set resources.computeDomains.enabled=true \
  --set resources.gpus.enabled=true
```

### Step 3: Build and Deploy GreenBoost Kernel Module

#### 3.1 Build Kernel Module on All Nodes

```bash
# Deploy container image with build tools
kubectl create configmap greenboost-module \
  --from-file=greenboost.c \
  --from-file=greenboost_ioctl.h

# Create DaemonSet to build module
kubectl apply -f k8s-deployment/greenboost-module-builder.yaml
```

#### 3.2 Load Kernel Module

```bash
# Verify module load on all nodes
kubectl get pods -n greenboost-system -l app=greenboost-module-loader

# Check module status on a node
kubectl exec -it <pod-name> -n greenboost-system -- lsmod | grep greenboost
```

### Step 4: Install GreenBoost DRA Driver (Helm Chart)

```bash
# Create greenboost namespace
kubectl create namespace greenboost-system

# Install GreenBoost DRA driver
helm install --wait --generate-name \
  deployments/helm/greenboost-dra-driver \
  --namespace greenboost-system \
  --values k8s-deployment/values-v100-cluster.yaml \
  --set greenboost.enable=true \
  --set greenboost.nvlinkPool=true \
  --set deviceClass.enabled=true \
  --set metricsExporter.enabled=true
```

### Step 5: Apply V100 Cluster Node Profile

```bash
# Copy profile to all nodes
kubectl create configmap v100-profile \
  --from-file=profiles/v100_cluster_node.md

# Apply profile via DaemonSet init container
kubectl apply -f k8s-deployment/greenboost-profile-loader.yaml
```

### Step 6: Verify Installation

#### 6.1 Check Kubelet Plugin Status

```bash
# Verify GreenBoost kubelet plugin DaemonSet pods
kubectl get pods -n greenboost-system -l component=plugin

# Expected output: 200/200 pods READY
```

#### 6.2 Verify GreenBoost Kernel Module

```bash
# Get module info from a node
kubectl exec -it <pod-name> -n greenboost-system \
  -- cat /sys/class/greenboost/greenboost/status

# Should show:
# Tier 1  GPU VRAM           : 256 GB
# Tier 2  System RAM pool    : 307 GB
# Tier 3  Lustre            : (dynamic)
```

#### 6.3 Verify DeviceClass

```bash
# Check GreenBoost device class
kubectl get deviceclass greenboost.nvidia.com -n greenboost-system

# Should show Ready status
```

#### 6.4 Verify Metrics Exporter

```bash
# Forward metrics exporter port
kubectl port-forward -n greenboost-system svc/greenboost-metrics 8080:8080

# Check metrics endpoint
curl http://localhost:8080/metrics | grep greenboost
```

### Step 7: Deploy Test Workload

```bash
# Deploy example LLM pod
kubectl apply -f k8s-examples/greenboost-llm-pod.yaml

# Monitor pod status
kubectl get pods -n greenboost-llm
kubectl logs -f llm-inference -n greenboost-llm -c ollama
```

### Step 8: Configure Monitoring

#### 8.1 Deploy Prometheus ServiceMonitor

```bash
kubectl apply -f k8s-deployment/servicemonitor.yaml
```

#### 8.2 Import Grafana Dashboard

1. Open Grafana UI
2. Navigate to Dashboards → Import
3. Paste dashboard JSON from `k8s-deployment/monitoring.md`
4. Select Prometheus data source
5. Save dashboard

#### 8.3 Verify Metrics

- Check "GreenBoost Cluster Memory" dashboard
- Verify T1/T2/T3 metrics appear
- Confirm NVLink health indicator

## Verification Checklist

- [ ] All 200 nodes have GreenBoost kernel module loaded
- [ ] All 200 kubelet plugin pods are Running
- [ ] DeviceClass `greenboost.nvidia.com` exists
- [ ] Metrics exporters are healthy on all nodes
- [ ] Prometheus is scraping GreenBoost metrics
- [ ] Grafana dashboard displays cluster aggregation
- [ ] Test LLM pod can allocate extended VRAM

## Troubleshooting

### Kernel Module Not Loading

```bash
# Check dmesg for errors
kubectl exec -it <pod-name> -- dmesg | grep greenboost

# Verify NVIDIA driver compatibility
kubectl exec -it <pod-name> -- nvidia-smi

# Check module parameters
kubectl exec -it <pod-name> -- modinfo greenboost
```

### Kubelet Plugin Not Starting

```bash
# Check plugin logs
kubectl logs -n greenboost-system <pod-name> -c greenboost-plugin

# Verify kubelet plugin directory
kubectl exec -it <pod-name> -- ls -la /var/lib/kubelet/plugins/
```

### Metrics Not Appearing

```bash
# Check ServiceMonitor status
kubectl get servicemonitor -n monitoring

# Verify Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# Open http://localhost:9090/targets
```

### NVLink Pooling Not Active

```bash
# Check NVLink fabric state
kubectl exec -it <pod-name> -- cat /sys/class/greenboost/greenboost/nvlink_ready

# Validate with NVIDIA DCGM
kubectl exec -it <pod-name> -- nvidia-smi nvlink
```

## Performance Tuning

### Memory Tier Sizing

For different cluster configurations, adjust in `values.yaml`:

```yaml
greenboost:
  physicalVramGb: 256   # 8× V100 32GB
  virtualVramGb: 307   # Adjust based on available system RAM
  tier3Backend: "lustre"
```

### Kubelet Plugin Optimization

```yaml
kubeletPlugin:
  resources:
    requests:
      cpu: "100m"
      memory: "256Mi"
    limits:
      cpu: "500m"
      memory: "512Mi"
```

## Scaling Deployment

### Multi-Architecture Support

For heterogeneous clusters (V100 + newer GPUs):

```yaml
nodeSelector:  # Remove selector for mixed cluster
  accelerator: nvidia-tesla-v100  # Only for V100 nodes
```

### Rolling Updates

```bash
# Update GreenBoost DRA driver
helm upgrade --install \
  deployments/helm/greenboost-dra-driver \
  --namespace greenboost-system \
  --values k8s-deployment/values-v100-cluster.yaml
```

## Security Considerations

### RBAC Configuration

Ensure minimal permissions for GreenBoost components:

```yaml
rbac:
  create: true
  rules:
  - apiGroups: [""]
    resources: ["nodes", "pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["resource.k8s.io"]
    resources: ["resourceclaims"]
    verbs: ["get", "list", "watch"]
```

### Network Policies

Restrict GreenBoost network access:

```bash
kubectl apply -f k8s-deployment/network-policy.yaml
```

## Next Steps

1. Deploy production LLM workloads
2. Configure automated scaling
3. Integrate with job schedulers (SLURM, PBS)
4. Set up alerting for critical events
5. Document cluster-specific customizations

## Support Resources

- [GreenBoost Documentation](README.md)
- [k8s-dra-driver-gpu Repository](https://github.com/NVIDIA/k8s-dra-driver-gpu)
- [Kubernetes DRA Documentation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
- [NVIDIA GPU Operator Guide](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)