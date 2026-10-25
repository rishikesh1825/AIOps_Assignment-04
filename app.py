"""
app.py - CPU-only image captioning (single image + bulk ZIP), instrumented for Prometheus.

Run:      streamlit run app.py
App UI:   http://localhost:8501
Metrics:  http://localhost:8001/metrics   (starts when the first browser tab opens the app)
"""
from __future__ import annotations

import io
import os
import time
import zipfile

import pandas as pd
import streamlit as st

import metrics as m
from captioner import (
    ALLOWED_EXTENSIONS,
    MAX_PIXELS,
    Captioner,
    InvalidImageError,
    load_image,
    make_thumbnail,
)

MAX_IMAGE_BYTES = 20 * 1024 * 1024            # per image
MAX_ZIP_IMAGES = 200                          # per ZIP
MAX_ZIP_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
GALLERY_LIMIT = 40

st.set_page_config(page_title="CPU Image Captioning", page_icon="🖼️", layout="wide")


# --------------------------------------------------------------------------- #
# One-time resources (shared by every session in this process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def start_metrics_server() -> bool:
    return m.start_metrics_server()


@st.cache_resource(show_spinner="Loading the captioning model (first run only)...")
def get_captioner() -> Captioner:
    try:
        captioner = Captioner()
    except Exception:
        m.record_model_failure()
        raise
    m.record_model_loaded(captioner)
    return captioner


# --------------------------------------------------------------------------- #
# Client metadata for custom labels
# --------------------------------------------------------------------------- #
def get_client() -> m.ClientInfo:
    session_id = "unknown"
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None:
            session_id = ctx.session_id[:8]
    except Exception:
        pass

    try:
        headers = st.context.headers
    except Exception:
        headers = {}

    ip = None
    try:
        ip = st.context.ip_address  # None when the app is opened via localhost
    except Exception:
        pass
    if not ip:
        forwarded = headers.get("X-Forwarded-For") or headers.get("X-Real-Ip")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        elif headers.get("Host", "").startswith(("localhost", "127.0.0.1")):
            ip = "127.0.0.1"
        else:
            ip = "unknown"

    return m.ClientInfo(session_id, ip, m.browser_family(headers.get("User-Agent", "")))


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def caption_image_bytes(captioner: Captioner, data: bytes, filename: str, mode: str) -> dict:
    """Caption one image and record its metrics. Never raises."""
    file_type = m.normalise_file_type(filename)
    row = {
        "file": filename, "caption": "", "status": "failed", "error": "",
        "latency_s": None, "size_kb": round(len(data) / 1024, 1), "resolution": "",
        "thumbnail": None,
    }

    start = time.perf_counter()
    try:
        image, info = load_image(data)
    except InvalidImageError as exc:
        m.record_image_failure(mode, file_type, len(data), exc.reason)
        row["error"] = str(exc)
        return row
    decode_s = time.perf_counter() - start

    try:
        result = captioner.caption(image)
    except Exception as exc:  # model errors must not crash the app
        m.record_image_failure(mode, file_type, len(data), "inference_error")
        row["error"] = f"Inference failed: {exc.__class__.__name__}"
        return row

    preprocess_s = decode_s + result.preprocess_s
    m.record_image_success(
        mode=mode, file_type=file_type, size_bytes=len(data),
        preprocess_s=preprocess_s, inference_s=result.inference_s, caption=result.text,
    )
    row.update(
        caption=result.text, status="success",
        latency_s=round(preprocess_s + result.inference_s, 2),
        resolution=f'{info["width"]}x{info["height"]}',
        thumbnail=make_thumbnail(image),
    )
    return row


def run_single(captioner: Captioner, uploaded, client: m.ClientInfo) -> dict:
    mode = "single"
    m.record_client(mode, client)
    start = time.perf_counter()
    with m.ACTIVE_REQUESTS.labels(mode).track_inprogress():
        row = caption_image_bytes(captioner, uploaded.getvalue(), uploaded.name, mode)
    m.record_request(mode, row["status"], time.perf_counter() - start)
    return row


def extract_zip(data: bytes) -> tuple[list[tuple[str, bytes]], list[dict]]:
    """Return (images, skipped). Raises zipfile.BadZipFile for a broken archive."""
    images, skipped = [], []
    total_bytes = 0

    def skip(name: str, reason: str) -> None:
        skipped.append({"file": name, "reason": reason})
        m.record_skipped(reason)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            base = os.path.basename(name)
            ext = os.path.splitext(base)[1].lower()
            if name.startswith("__MACOSX/") or base.startswith("."):
                skip(name, "hidden_file")
            elif ext not in ALLOWED_EXTENSIONS:
                skip(name, "not_image")
            elif info.file_size > MAX_IMAGE_BYTES:
                skip(name, "too_large")
            elif len(images) >= MAX_ZIP_IMAGES or total_bytes + info.file_size > MAX_ZIP_UNCOMPRESSED_BYTES:
                skip(name, "limit_exceeded")
            else:
                try:
                    images.append((name, archive.read(info)))
                    total_bytes += info.file_size
                except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError):
                    skip(name, "unreadable")  # e.g. encrypted or damaged entry

    return images, skipped


def run_bulk(captioner: Captioner, uploaded, client: m.ClientInfo, progress, status_box) -> dict:
    mode = "bulk"
    m.record_client(mode, client)
    start = time.perf_counter()
    result = {"name": uploaded.name, "rows": [], "skipped": [], "error": "", "duration_s": 0.0}

    with m.ACTIVE_REQUESTS.labels(mode).track_inprogress():
        try:
            images, result["skipped"] = extract_zip(uploaded.getvalue())
        except zipfile.BadZipFile:
            m.ERRORS.labels(mode, "bad_zip").inc()
            result["error"] = "This file is not a valid ZIP archive."
            m.record_request(mode, "failed", time.perf_counter() - start)
            return result

        total = len(images)
        if total == 0:
            m.ERRORS.labels(mode, "empty_zip").inc()
            result["error"] = "No usable images were found in this ZIP."
            m.record_request(mode, "failed", time.perf_counter() - start)
            return result

        try:
            for i, (name, data) in enumerate(images, start=1):
                m.BULK_QUEUE_REMAINING.set(total - i + 1)
                status_box.caption(f"Captioning {i}/{total}: {name}")
                result["rows"].append(caption_image_bytes(captioner, data, name, mode))
                progress.progress(i / total)
        finally:
            m.BULK_QUEUE_REMAINING.set(0)

    ok = sum(r["status"] == "success" for r in result["rows"])
    status = "success" if ok == total else ("partial" if ok else "failed")
    result["duration_s"] = time.perf_counter() - start
    m.record_request(mode, status, result["duration_s"])
    return result


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_sidebar(captioner: Captioner, client: m.ClientInfo, server_started: bool) -> None:
    with st.sidebar:
        st.header("Runtime")
        st.markdown(f"**Model:** `{captioner.model_name}`")
        st.markdown(f"**Device:** CPU · {captioner.torch_threads} threads")
        st.markdown(f"**Model load time:** {captioner.load_seconds:.1f} s")
        st.markdown(f"**Parameters:** {captioner.num_parameters / 1e6:.0f} M")
        st.divider()
        st.header("Monitoring")
        state = "running" if server_started else "already running / port busy"
        st.markdown(f"**Metrics:** `http://localhost:{m.METRICS_PORT}/metrics` ({state})")
        st.markdown(f"**Your session:** `{client.session_id}`")
        st.markdown(f"**Your IP:** `{client.ip}` · **Browser:** {client.browser}")
        st.divider()
        st.caption(
            f"Limits: {MAX_IMAGE_BYTES // 2**20} MB per image, {MAX_PIXELS // 10**6} MP max, "
            f"{MAX_ZIP_IMAGES} images per ZIP."
        )


def render_single(captioner: Captioner, client: m.ClientInfo) -> None:
    st.subheader("Caption a single image")
    uploaded = st.file_uploader(
        "Choose an image",
        type=sorted({e.lstrip(".") for e in ALLOWED_EXTENSIONS}),
        key="single_upload",
    )
    if uploaded is None:
        st.session_state.pop("single_result", None)
        return

    left, right = st.columns(2)
    with left:
        try:
            st.image(uploaded, width=420)
        except Exception:
            st.warning("Preview unavailable - this file may not be a valid image.")
    with right:
        if st.button("Generate caption", type="primary", key="single_go"):
            with st.spinner("Generating caption on CPU..."):
                row = run_single(captioner, uploaded, client)
            st.session_state["single_result"] = (uploaded.file_id, row)

        saved = st.session_state.get("single_result")
        if saved and saved[0] == uploaded.file_id:
            row = saved[1]
            if row["status"] == "success":
                st.success(f"**{row['caption']}**")
                st.caption(f"{row['latency_s']} s · {row['resolution']} · {row['size_kb']} KB")
            else:
                st.error(row["error"])


def render_bulk(captioner: Captioner, client: m.ClientInfo) -> None:
    st.subheader("Caption a ZIP of images")
    uploaded = st.file_uploader("Choose a ZIP file", type=["zip"], key="bulk_upload")
    if uploaded is not None and st.button("Caption all images", type="primary", key="bulk_go"):
        progress = st.progress(0.0)
        status_box = st.empty()
        st.session_state["bulk_result"] = run_bulk(captioner, uploaded, client, progress, status_box)
        status_box.empty()

    result = st.session_state.get("bulk_result")
    if not result:
        return
    if result["error"]:
        st.error(result["error"])
        if result["skipped"]:
            st.dataframe(pd.DataFrame(result["skipped"]), hide_index=True)
        return

    rows = result["rows"]
    ok = [r for r in rows if r["status"] == "success"]
    avg = sum(r["latency_s"] for r in ok) / len(ok) if ok else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images found", len(rows))
    c2.metric("Captioned", len(ok))
    c3.metric("Failed", len(rows) - len(ok))
    c4.metric("Skipped files", len(result["skipped"]))
    c5.metric("Avg latency / image", f"{avg:.2f} s")
    st.caption(f"{result['name']} · total time {result['duration_s']:.1f} s")

    table = pd.DataFrame(rows).drop(columns=["thumbnail"])
    st.dataframe(table, hide_index=True)
    st.download_button(
        "Download captions (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{os.path.splitext(result['name'])[0]}_captions.csv",
        mime="text/csv",
    )

    if result["skipped"]:
        with st.expander(f"Skipped files ({len(result['skipped'])})"):
            st.dataframe(pd.DataFrame(result["skipped"]), hide_index=True)

    with st.expander("Gallery", expanded=True):
        cols = st.columns(4)
        for i, row in enumerate(ok[:GALLERY_LIMIT]):
            with cols[i % 4]:
                st.image(row["thumbnail"], caption=row["caption"], width=220)
        if len(ok) > GALLERY_LIMIT:
            st.caption(f"Showing the first {GALLERY_LIMIT} of {len(ok)} images.")


def main() -> None:
    server_started = start_metrics_server()
    captioner = get_captioner()
    client = get_client()

    st.title("🖼️ CPU Image Captioning")
    st.caption("BLIP running on CPU · instrumented with Prometheus")
    render_sidebar(captioner, client, server_started)

    single_tab, bulk_tab = st.tabs(["Single image", "Bulk (ZIP)"])
    with single_tab:
        render_single(captioner, client)
    with bulk_tab:
        render_bulk(captioner, client)


main()
