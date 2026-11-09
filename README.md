# A Lightweight Digital Twin Framework for Real-Time DDoS Detection in Edge-IoT Networks

Official implementation and experimental benchmark repository for the manuscript:
> **"A Lightweight Digital Twin Framework for Real-Time DDoS Detection in Edge-IoT Networks"**  
> *Oktay Kara\*, İlker Türker, İsa Avcı*  
> Department of Computer Engineering, Karabük University, Karabük 78050, Türkiye

---

## 1. Overview

This repository provides an edge-centric **Digital Twin (DT)** intrusion detection framework designed for resource-constrained Industrial Internet of Things (IIoT) gateways (2 vCPU cores, 2 GB RAM). Key architectural contributions include:

- **Event-Driven Digital Twin State Synchronization ($\varepsilon_{sync} = 0.05$):** Continuously tracks the $L_2$ divergence between physical flow states ($V_p$) and virtual replica states ($V_v$), bypassing redundant classifier invocations during steady-state traffic ($\varepsilon_{sync} \le 0.05$) and achieving sub-millisecond synchronization ($T_{sync} = 0.65\text{ ms}$) within an $18.5\text{ ms}$ end-to-end micro-batch detection pipeline ($T_{proc} = 17.10\text{ ms}$, $T_{inf} = 0.75\text{ ms}$).
- **Port-Free Dual-Tier Hybrid Feature Engineering (IG-PCA):** Purges static network identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`) and transport ports (`Source Port`, `Destination Port`, plus duplicate `Fwd Header Length.1`) to prevent empirical data leakage. From the standardized **78-dimensional port-free feature space**, 8 high-entropy flow/packet headers (`Protocol`, `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, `Flow Packets/s`, `Flow Bytes/s`, `Average Packet Size`, `Inbound`) are selected via Information Gain (IG), while the 70 residual attributes are projected onto 7 orthogonal Principal Components (`PC1`–`PC7`, retaining **98.09%** variance), yielding a compact **15-dimensional hybrid vector** (**80.7%** dimensionality reduction and **~82%** reduction in per-flow FLOPs).
- **8-Bit Structural Random Forest Quantization:** Discretizes split thresholds, node offsets, and terminal leaf distributions across 50 trees (max depth 15, 130,760 total nodes) into 8-bit integer representations, compressing the model structure by **77.2%** (from **5.49 MB** to **1.25 MB**) while maintaining an active streaming container footprint of **206.8 MB** RAM and **11.6%** CPU load.

---

## 2. Repository Structure

```text
├── data/
│   ├── CICDDoS2019_sample.csv          # Stratified 99,998-flow benchmark subset for 5-Fold CV
│   └── ablation_study_table7.csv       # 6-configuration 5-Fold CV ablation results (Table 7)
├── figures/
│   ├── create_figure_*.py              # High-resolution (600 DPI) figure generation scripts
│   └── figure1.png .. figure8_swot.png # Manuscript figures
├── models/
│   ├── scaler.joblib                   # Min-Max normalization scaler (78 port-free features)
│   ├── pca_model.joblib                # 7-component PCA projection model (on 70 residuals)
│   ├── selected_features.joblib        # Standardized 78 port-free feature names
│   ├── label_encoder.joblib            # 4-category operational threat encoder
│   ├── rf_model.joblib                 # Trained Random Forest classifier (15 hybrid features)
│   └── model_quantized.tflite          # Structural 8-bit quantized edge inference artifact
├── src/
│   ├── 0_topology.py                   # Mininet-WiFi 50-node IoT + 5 AP edge network topology
│   ├── 1_data_prep.py                  # CICDDoS2019 4-category operational attack mapping
│   ├── 2_feature_eng.py                # Port-free 78 -> 15 Hybrid IG-PCA feature engineering
│   ├── 3_model_quant.py                # Random Forest training & structural quantization
│   ├── 4_metric_evaluation.py          # Hold-out test evaluation (Accuracy, Weighted/Macro F1, CM)
│   ├── 5_monitoring_engine.py          # Event-driven Digital Twin synchronization engine
│   ├── 6_result_visualization.py       # Latency & CPU utilization visualization
│   ├── 7_pcap_evaluation.py            # Streaming PCAP ingestion & flow analysis
│   ├── 8_performance_proof.py          # End-to-end latency breakdown profiler (T_proc, T_sync, T_inf)
│   └── 9_ablation_experiments.py       # 6-configuration 5-Fold Stratified CV ablation benchmark
├── Dockerfile                          # Ubuntu 22.04 + Python 3.11 edge container definition
├── docker-compose.yml                  # 2.0 vCPU / 2 GB RAM cgroup-constrained edge deployment
└── requirements.txt                    # Python dependencies
```

---

## 3. Experimental Summary

### 3.1. Hold-Out Test Set Performance (285,747 Disjoint Flows — Port-Free)

| Operational Category | CICDDoS2019 Vectors | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BENIGN (Normal)** | Normal traffic (HTTP, HTTPS, etc.) | 0.9897 (0.99) | 0.9934 (0.99) | **0.9916 (0.99)** | 1,831 |
| **Reflection-Based** | LDAP, MSSQL, NetBIOS, Portmap, TFTP | 0.9632 (0.96) | 0.9324 (0.93) | **0.9475 (0.95)** | 139,926 |
| **TCP-Based** | SYN Flood, WebDDoS | 0.9990 (1.00) | 0.9991 (1.00) | **0.9990 (1.00)** | 40,048 |
| **UDP-Based** | UDP, UDP-Lag, NTP, DNS, SSDP, SNMP | 0.9125 (0.91) | 0.9516 (0.95) | **0.9316 (0.93)** | 103,942 |
| **OVERALL / TOTAL** | **4 Consolidated Categories** | **Weighted: 94.99%**<br>**Macro: 96.61%** | **Accuracy: 94.91%**<br>**Macro: 96.91%** | **Weighted F1: 94.92%**<br>**Macro F1: 96.74%** | **285,747** |

### 3.2. 6-Configuration Component Ablation Study (5-Fold Stratified CV — 99,998 Flows)

| Configuration / Model Architecture | Dim | 5-Fold CV Accuracy (%) | 5-Fold CV Weighted / Macro F1 (%) | Per-Flow Inference ($T_{inf}$) | Model Size / Batch RSS (MB) | Pipeline CPU Load (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **(1) Raw Feature Baseline (Float32 RF)** | 78 | 94.77 ± 0.19% | 94.79 ± 0.19% / 96.35 ± 0.20% | 1.2 ± 0.3 µs (0.0012 ms) | 5.82 MB / 442.4 MB | 85.9 ± 0.9% |
| **(2) IG Domain Headers Alone (Port-Free)** | 8 | 94.57 ± 0.13% | 94.58 ± 0.13% / 95.75 ± 0.13% | 0.9 ± 0.3 µs (0.0009 ms) | 4.95 MB / 389.9 MB | 10.6 ± 0.2% |
| **(3) Standard PCA Alone** | 15 | 94.84 ± 0.17% | 94.85 ± 0.17% / 96.24 ± 0.33% | 0.9 ± 0.2 µs (0.0009 ms) | 5.12 MB / 396.7 MB | 13.9 ± 0.4% |
| **(4) Proposed Hybrid IG-PCA (Float32 RF)** | 15 | 94.84 ± 0.15% | 94.85 ± 0.15% / 96.17 ± 0.13% | 0.7 ± 0.0 µs (0.0007 ms) | 5.49 MB / 440.0 MB | 17.9 ± 0.6% |
| **(5) Proposed Full Framework (Quantized RF + DT)** | **15** | **94.85 ± 0.15%** | **94.86 ± 0.15% / 96.18 ± 0.14%** | **0.7 ± 0.0 µs (0.0007 ms)** | **1.25 MB / 432.8 MB (206.8 MB stream)** | **11.6 ± 0.5%** |
| **(6) DT-less Baseline (Equivalent Quantized IDS)** | 15 | 94.84 ± 0.15% | 94.85 ± 0.15% / 96.17 ± 0.13% | 0.6 ± 0.2 µs (0.0006 ms) | 1.25 MB / 445.3 MB (383.8 MB stream) | 76.4 ± 0.7% |

### 3.3. End-to-End Streaming Detection Latency Breakdown (500-Flow Micro-Batches, 2-vCPU Limit)

| Pipeline Stage / Metric | Formulation Symbol | Measured Latency (2-vCPU Edge Target) | Share of Total $T_{det}$ |
| :--- | :---: | :---: | :---: |
| **Streaming Feature Extraction & Hybrid IG-PCA Projection** | $T_{proc}$ | **17.10 ms** | 92.4% |
| **Event-Driven Digital Twin State Synchronization** | $T_{sync}$ | **0.65 ms (± 0.04 ms)** | 3.5% |
| **Quantized 8-Bit RF Micro-Batch Inference (500 flows)** | $T_{inf}$ | **0.75 ms** *(500 × 0.7 µs + 0.40 ms wrapper)* | 4.1% |
| **Total End-to-End Detection Latency** | $T_{det} = T_{sync} + T_{proc} + T_{inf}$ | **18.50 ms** *(68.1% faster than 58.0 ms standard edge; 88.1% faster than 155.0 ms cloud)* | **100.0%** |

---

## 4. Quick Start & Reproducing Experiments

### 4.1. Environment Setup
```bash
pip install -r requirements.txt
```

### 4.2. Running the 5-Fold Cross-Validation Ablation Benchmark (Out-of-the-Box)
The repository includes `data/CICDDoS2019_sample.csv` (99,998 real flows), allowing immediate execution of the 6-configuration 5-fold cross-validation ablation study:
```bash
python src/9_ablation_experiments.py
```

### 4.3. Running the Latency Breakdown Profiler
```bash
python src/8_performance_proof.py
```

### 4.4. Full Pipeline Execution (With Full CICDDoS2019 Parquet Partitions)
If you place the full CICDDoS2019 parquet files in `data/processed/`, you can run the complete training and evaluation pipeline sequentially:
```bash
python src/1_data_prep.py
python src/2_feature_eng.py
python src/3_model_quant.py
python src/4_metric_evaluation.py
python src/5_monitoring_engine.py
```

### 4.5. Containerized Edge Gateway Emulation (2 vCPU, 2 GB RAM)
```bash
docker compose up --build
```

---

## 5. License & Contact
For academic inquiries, please contact the corresponding author: **Oktay Kara** (`2528126003@ogrenci.karabuk.edu.tr`).
