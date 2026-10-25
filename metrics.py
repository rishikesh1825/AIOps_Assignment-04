"""
metrics.py - the 20 Prometheus metrics exposed by the image-captioning app.

Why a separate module?
    Streamlit re-executes app.py top-to-bottom on every click, but Python
    modules are imported only once per process. Defining the metrics here
    means each one is registered exactly once. `_get_or_create` additionally
    survives Streamlit's hot-reload (it re-imports a module when you save it),
    which would otherwise raise "Duplicated timeseries in CollectorRegistry".

Naming follows Prometheus conventions:
    imgcap_<what>_<base unit>, counters end in _total, units are seconds/bytes.

Metric types used (6 counters, 7 gauges, 4 histograms, 2 summaries, 1 info)
    Counter   - monotonically increasing totals          (exposed as *_total)
    Gauge     - values that go up and down               (current state)
    Histogram - distributions with buckets               (*_bucket, *_sum, *_count)
    Summary   - running count + sum (no buckets)         (*_sum, *_count)
    Info      - static key/value build information       (*_info)

The prometheus_client default registry also exports process_* metrics
(e.g. process_start_time_seconds, used for uptime on the dashboard).
"""
from __future__ import annotations

import os
import platform
import threading
import time
from typing import NamedTuple

import psutil
from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    start_http_server,
)

try:  # drop the extra *_created series so /metrics stays readable
    from prometheus_client import disable_created_metrics

    disable_created_metrics()
except ImportError:  # very old prometheus_client
    pass

NAMESPACE = "imgcap"
APP_VERSION = "1.0.0"
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))
SESSION_WINDOW_SECONDS = 300  # a session is "active" if seen in the last 5 minutes

KNOWN_FILE_TYPES = {"jpg", "png", "bmp", "gif", "webp", "tif"}


# --------------------------------------------------------------------------- #
# Registration helpers
# --------------------------------------------------------------------------- #
def _get_or_create(cls, name: str, documentation: str, labelnames=(), **kwargs):
    """Return the already-registered collector after a hot-reload, else create it."""
    existing = REGISTRY._names_to_collectors.get(f"{NAMESPACE}_{name}")  # noqa: SLF001
    if existing is not None:
        return existing
    return cls(name, documentation, labelnames=labelnames, namespace=NAMESPACE, **kwargs)


def _shared_state() -> dict:
    """State that must outlive module reloads (attached to the global registry)."""
    state = getattr(REGISTRY, "_imgcap_state", None)
    if state is None:
        state = {
            "sessions": {},               # session_id -> last-seen unix time
            "lock": threading.Lock(),
            "process": psutil.Process(),
        }
        REGISTRY._imgcap_state = state  # noqa: SLF001
    return state


_STATE = _shared_state()

# Bucket layouts (seconds / bytes) chosen for a CPU-bound BLIP model.
INFERENCE_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 20, 30)
PREPROCESS_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2)
REQUEST_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200)
SIZE_BUCKETS = (10e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2.5e6, 5e6, 10e6, 25e6)


# --------------------------------------------------------------------------- #
# Counters (6) - always increase; query them with rate() / increase()
# --------------------------------------------------------------------------- #
REQUESTS = _get_or_create(
    Counter, "requests_total",
    "Caption requests handled. One per single-image upload or per ZIP job.",
    ["mode", "status"],               # mode: single|bulk  status: success|partial|failed
)
IMAGES_PROCESSED = _get_or_create(
    Counter, "images_processed_total",
    "Images that went through the captioning pipeline.",
    ["mode", "status", "file_type"],  # status: success|failed
)
ERRORS = _get_or_create(
    Counter, "errors_total",
    "Errors while handling uploads, by error type.",
    ["mode", "error_type"],           # corrupt_image|too_large|inference_error|bad_zip|empty_zip
)
SKIPPED_FILES = _get_or_create(
    Counter, "zip_skipped_files_total",
    "Files inside ZIP uploads that were skipped before captioning.",
    ["reason"],                       # not_image|hidden_file|too_large|limit_exceeded|unreadable
)
CLIENT_REQUESTS = _get_or_create(
    Counter, "client_requests_total",
    "Requests per client. Custom labels identify who is using the app.",
    ["session_id", "client_ip", "browser", "mode"],
)
CAPTION_WORDS = _get_or_create(
    Counter, "caption_words_total",
    "Total words generated across all captions (pairs with caption_length_words_sum).",
    ["mode"],
)


# --------------------------------------------------------------------------- #
# Gauges (7) - current values; can go up and down
# --------------------------------------------------------------------------- #
ACTIVE_REQUESTS = _get_or_create(
    Gauge, "active_requests", "Requests currently being processed.", ["mode"],
)
ACTIVE_SESSIONS = _get_or_create(
    Gauge, "active_sessions",
    f"Browser sessions that made a request in the last {SESSION_WINDOW_SECONDS}s.",
)
BULK_QUEUE_REMAINING = _get_or_create(
    Gauge, "bulk_queue_remaining", "Images still waiting in the running bulk job.",
)
MODEL_LOADED = _get_or_create(
    Gauge, "model_loaded", "1 if the captioning model is loaded and ready, else 0.",
)
MODEL_MEMORY_BYTES = _get_or_create(
    Gauge, "model_memory_bytes", "Memory held by model weights and buffers (bytes).",
)
PROCESS_RSS_BYTES = _get_or_create(
    Gauge, "process_memory_rss_bytes", "Resident memory of the app process (psutil).",
)
PROCESS_CPU_PERCENT = _get_or_create(
    Gauge, "process_cpu_percent",
    "CPU used by the app process as % of total machine capacity (0-100).",
)


# --------------------------------------------------------------------------- #
# Histograms (4) - distributions; use histogram_quantile() for p50/p95/p99
# --------------------------------------------------------------------------- #
INFERENCE_LATENCY = _get_or_create(
    Histogram, "inference_latency_seconds",
    "Model generate() time per image.",
    ["mode"], buckets=INFERENCE_BUCKETS,
)
PREPROCESS_LATENCY = _get_or_create(
    Histogram, "preprocess_latency_seconds",
    "Decode + validation + processor time per image.",
    ["mode"], buckets=PREPROCESS_BUCKETS,
)
REQUEST_DURATION = _get_or_create(
    Histogram, "request_duration_seconds",
    "End-to-end time per request (one image, or a whole ZIP job).",
    ["mode"], buckets=REQUEST_BUCKETS,
)
IMAGE_SIZE = _get_or_create(
    Histogram, "image_size_bytes",
    "Size of uploaded image files.",
    ["mode"], buckets=SIZE_BUCKETS,
)


# --------------------------------------------------------------------------- #
# Summaries (2) - exposed as *_sum and *_count; average = sum / count
# --------------------------------------------------------------------------- #
CAPTION_LENGTH = _get_or_create(
    Summary, "caption_length_words", "Words per generated caption.", ["mode"],
)
IMAGE_PROCESSING = _get_or_create(
    Summary, "image_processing_seconds",
    "Total processing time per image (preprocess + inference).", ["mode"],
)


# --------------------------------------------------------------------------- #
# Info (1) - static labels about the build, exposed as imgcap_build_info
# --------------------------------------------------------------------------- #
BUILD_INFO = _get_or_create(Info, "build", "Application and runtime build information.")


# --------------------------------------------------------------------------- #
# Callback gauges - evaluated at scrape time, so they are always fresh
# --------------------------------------------------------------------------- #
def _count_active_sessions() -> int:
    cutoff = time.time() - SESSION_WINDOW_SECONDS
    with _STATE["lock"]:
        stale = [sid for sid, seen in _STATE["sessions"].items() if seen < cutoff]
        for sid in stale:
            del _STATE["sessions"][sid]
        return len(_STATE["sessions"])


def _initialise_label_sets() -> None:
    """Pre-create known label combinations at 0.

    A labelled counter series only exists after its first .inc(), and it then
    starts at 1 - so rate()/increase() never see that first event. Creating the
    expected series up front at 0 makes the very first error or request visible
    on the dashboard. (client_requests_total labels are dynamic, so it is skipped.)
    """
    statuses = {"single": ("success", "failed"), "bulk": ("success", "partial", "failed")}
    errors = {
        "single": ("corrupt_image", "too_large", "inference_error"),
        "bulk": ("corrupt_image", "too_large", "inference_error", "bad_zip", "empty_zip"),
    }
    for mode in ("single", "bulk"):
        for status in statuses[mode]:
            REQUESTS.labels(mode, status)
        for status in ("success", "failed"):
            for file_type in (*sorted(KNOWN_FILE_TYPES), "other"):
                IMAGES_PROCESSED.labels(mode, status, file_type)
        for error_type in errors[mode]:
            ERRORS.labels(mode, error_type)
        CAPTION_WORDS.labels(mode)
        ACTIVE_REQUESTS.labels(mode)
        for histogram in (INFERENCE_LATENCY, PREPROCESS_LATENCY, REQUEST_DURATION, IMAGE_SIZE):
            histogram.labels(mode)
        CAPTION_LENGTH.labels(mode)
        IMAGE_PROCESSING.labels(mode)
    for reason in ("not_image", "hidden_file", "too_large", "limit_exceeded", "unreadable"):
        SKIPPED_FILES.labels(reason)


_initialise_label_sets()

_CPU_COUNT = psutil.cpu_count() or 1
PROCESS_RSS_BYTES.set_function(lambda: _STATE["process"].memory_info().rss)
PROCESS_CPU_PERCENT.set_function(lambda: _STATE["process"].cpu_percent(None) / _CPU_COUNT)
ACTIVE_SESSIONS.set_function(_count_active_sessions)
if not _STATE.get("build_info_set"):  # don't overwrite the model name after a hot-reload
    BUILD_INFO.info({
        "version": APP_VERSION,
        "python_version": platform.python_version(),
        "device": "cpu",
        "model": "not_loaded",
    })


# --------------------------------------------------------------------------- #
# Public helpers used by app.py
# --------------------------------------------------------------------------- #
class ClientInfo(NamedTuple):
    session_id: str
    ip: str
    browser: str


def start_metrics_server(port: int = METRICS_PORT) -> bool:
    """Start the /metrics HTTP server in a background thread. Returns False if the port is taken."""
    try:
        start_http_server(port)
        return True
    except OSError:
        # Already running in this process (Streamlit hot-reload) or port in use.
        return False


def browser_family(user_agent: str) -> str:
    """Collapse a User-Agent string into a small, fixed set of label values."""
    ua = (user_agent or "").lower()
    if not ua:
        return "unknown"
    for token, family in (("edg/", "edge"), ("opr/", "opera"), ("firefox", "firefox"),
                          ("chrome", "chrome"), ("safari", "safari"),
                          ("python", "script"), ("curl", "script")):
        if token in ua:
            return family
    return "other"


def normalise_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    ext = {"jpeg": "jpg", "tiff": "tif"}.get(ext, ext)
    return ext if ext in KNOWN_FILE_TYPES else "other"


def record_client(mode: str, client: ClientInfo) -> None:
    CLIENT_REQUESTS.labels(client.session_id, client.ip, client.browser, mode).inc()
    with _STATE["lock"]:
        _STATE["sessions"][client.session_id] = time.time()


def record_image_success(mode: str, file_type: str, size_bytes: int,
                         preprocess_s: float, inference_s: float, caption: str) -> None:
    words = len(caption.split())
    IMAGES_PROCESSED.labels(mode, "success", file_type).inc()
    IMAGE_SIZE.labels(mode).observe(size_bytes)
    PREPROCESS_LATENCY.labels(mode).observe(preprocess_s)
    INFERENCE_LATENCY.labels(mode).observe(inference_s)
    IMAGE_PROCESSING.labels(mode).observe(preprocess_s + inference_s)
    CAPTION_LENGTH.labels(mode).observe(words)
    CAPTION_WORDS.labels(mode).inc(words)


def record_image_failure(mode: str, file_type: str, size_bytes: int, reason: str) -> None:
    IMAGES_PROCESSED.labels(mode, "failed", file_type).inc()
    IMAGE_SIZE.labels(mode).observe(size_bytes)
    ERRORS.labels(mode, reason).inc()


def record_request(mode: str, status: str, duration_s: float) -> None:
    REQUESTS.labels(mode, status).inc()
    REQUEST_DURATION.labels(mode).observe(duration_s)


def record_skipped(reason: str) -> None:
    SKIPPED_FILES.labels(reason).inc()


def record_model_loaded(captioner) -> None:
    MODEL_LOADED.set(1)
    MODEL_MEMORY_BYTES.set(captioner.memory_bytes)
    BUILD_INFO.info({
        "version": APP_VERSION,
        "python_version": platform.python_version(),
        "device": "cpu",
        "model": captioner.model_name,
        "torch_version": captioner.torch_version,
    })
    _STATE["build_info_set"] = True


def record_model_failure() -> None:
    MODEL_LOADED.set(0)
