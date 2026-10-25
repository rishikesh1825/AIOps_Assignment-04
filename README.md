# Image Captioning with Prometheus & Grafana Monitoring

A Streamlit app that captions images on the CPU with BLIP (`Salesforce/blip-image-captioning-base`),
for a single image or a ZIP of images. It exposes Prometheus metrics, which Prometheus scrapes
together with node_exporter, alerts on through Alertmanager, and Grafana visualises on a dashboard.

```
Browser ─► Streamlit app :8501 ─► /metrics :8001 ─┐
                                                   ├─► Prometheus :9090 ─► Grafana :3000
node_exporter :9100 ───────────────────────────────┘        │
                                                             └─► Alertmanager :9093 ─► alert_receiver.py :5001
```

---

## Repository structure

```
Assignment-04/
├── app.py                      # Streamlit UI and request pipeline for single-image and bulk (ZIP) captioning.
├── captioner.py                # Loads the BLIP model on CPU and safely decodes and validates uploaded images.
├── metrics.py                  # Defines all 20 Prometheus metrics and the helpers that record them.
├── requirements.txt            # Python dependencies (CPU-only PyTorch).
├── AI_PROMPTS.md               # Records where AI was used and explains each PromQL query by hand.
├── README.md                   # This guide.
├── .gitignore                  # Excludes the virtualenv, Prometheus/Alertmanager data, raw images and videos.
│
├── .streamlit/
│   └── config.toml             # Raises the upload limit to 200 MB and disables Streamlit usage stats.
│
├── prometheus/
│   ├── prometheus.yml          # Scrapes the app (:8001), node_exporter (:9100) and Prometheus every 5 s and sends alerts to Alertmanager.
│   └── alert_rules.yml         # Nine alert rules for app health, latency, failure rate, memory, CPU and disk.
│
├── alertmanager/
│   ├── alertmanager.yml        # Groups alerts, routes them by severity and mutes dependent alerts.
│   └── alert_receiver.py       # Webhook endpoint that prints each alert notification and logs it to alerts.log.
│
├── grafana/
│   ├── dashboard.json          # The 16-panel Grafana dashboard (classic export, imports into any Grafana version).
│   └── dashboard_v2.json       # The same dashboard in Grafana 12's v2 export format.
│
├── sample_data/
│   ├── normal.zip              # Clean images for trying bulk mode.
│   └── messy.zip               # Two images plus a text file and a fake .jpg, to show error handling.
│
└── screenshots/                # Screenshots of each component running.
```

---

## Prerequisites

- Ubuntu (or another Linux) with Python 3.10+
- [Prometheus](https://prometheus.io/download/) 3.x, node_exporter 1.x and Alertmanager 0.2x+ (Linux amd64 tarballs)
- Docker (for Grafana)
- About 3 GB free disk space (PyTorch and the ~1 GB BLIP model)

---

## Setup

```bash
# 1. Get the code
git clone <repo-url> Assignment-04
cd Assignment-04

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) Download the BLIP model now so the first app start is fast
python -c "from transformers import BlipProcessor, BlipForConditionalGeneration as M; \
n='Salesforce/blip-image-captioning-base'; BlipProcessor.from_pretrained(n); M.from_pretrained(n)"
```

---

## Running

Run each component in its own terminal **from the repository root**. Replace `/path/to/` with the
folders where the Prometheus, node_exporter and Alertmanager tarballs were extracted.

**1. node_exporter** (system CPU, memory and disk metrics)
```bash
/path/to/node_exporter-*/node_exporter
```

**2. Prometheus**
```bash
/path/to/prometheus-*/promtool check config prometheus/prometheus.yml     # should print SUCCESS
/path/to/prometheus-*/prometheus --config.file=prometheus/prometheus.yml --storage.tsdb.retention.size=500MB
```

**3. The captioning app**
```bash
source .venv/bin/activate
streamlit run app.py
```
Open http://localhost:8501. The metrics endpoint (http://localhost:8001/metrics) starts the first
time the page is opened, and the model loads on that first visit.

**4. Alert receiver and Alertmanager**
```bash
source .venv/bin/activate
python alertmanager/alert_receiver.py
```
```bash
/path/to/alertmanager-*/amtool check-config alertmanager/alertmanager.yml
/path/to/alertmanager-*/alertmanager --config.file=alertmanager/alertmanager.yml \
    --storage.path=alertmanager/data --cluster.listen-address=""
```

**5. Grafana**
```bash
# First time: create the container
docker run -d -p 3000:3000 --name=grafana-a4 \
    --add-host=host.docker.internal:host-gateway grafana/grafana-enterprise

# Later runs
docker start grafana-a4
```
Then, at http://localhost:3000 (login `admin` / `admin`):
1. **Connections → Data sources → Add new data source → Prometheus**, URL
   `http://host.docker.internal:9090`, then **Save & test**.
2. **Dashboards → New → Import**, upload `grafana/dashboard.json`, select the Prometheus data source,
   then **Import**.

### Check that everything is running

| Component | URL | Expected |
|---|---|---|
| App | http://localhost:8501 | Sidebar shows the model loaded |
| App metrics | http://localhost:8001/metrics | Lines starting with `imgcap_` |
| node_exporter | http://localhost:9100/metrics | Lines starting with `node_` |
| Prometheus | http://localhost:9090/targets | `caption_app`, `node`, `prometheus` all **UP** |
| Alertmanager | http://localhost:9093 | Alertmanager UI loads |
| Grafana | http://localhost:3000 | Dashboard health row shows ✓ UP |

---

## Using the app

- **Single image tab:** upload an image and click **Generate caption**.
- **Bulk (ZIP) tab:** upload a ZIP (e.g. `sample_data/normal.zip`) and click **Caption all images**
  to get a results table, a CSV download and a gallery. Corrupt files, non-images and oversized
  files are skipped or reported rather than crashing the app (try `sample_data/messy.zip`).
- Watch the Grafana dashboard while a bulk job runs: the job appears as a shaded band, and
  throughput, latency, app CPU and host CPU rise together.

---

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `METRICS_PORT` | `8001` | Port of the app's `/metrics` endpoint (update `prometheus/prometheus.yml` if changed) |
| `CAPTION_MODEL` | `Salesforce/blip-image-captioning-base` | Hugging Face captioning model |
| `CAPTION_MAX_TOKENS` | `30` | Maximum caption length in tokens |
| `TORCH_THREADS` | PyTorch default | CPU threads used for inference |
| `ALERT_RECEIVER_PORT` | `5001` | Port of the alert webhook receiver |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `localhost:8001/metrics` doesn't load | Open the app at http://localhost:8501 once; the metrics server starts on the first page load. |
| Port 8001 already in use | Start the app with `METRICS_PORT=8002 streamlit run app.py` and change the target in `prometheus/prometheus.yml`. |
| Grafana data source test fails | Inside Docker, `localhost` is the container; use `http://host.docker.internal:9090`. |
| Dashboard import fails on an older Grafana | Import `grafana/dashboard.json`, not `dashboard_v2.json`. |
| Alertmanager exits with "no private IP found" | Keep the `--cluster.listen-address=""` flag. |
| An alert stays "pending" | Each rule has a `for:` duration (up to 5 minutes) before it fires. |
