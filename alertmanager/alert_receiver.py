"""
alert_receiver.py - a tiny webhook endpoint that Alertmanager sends notifications to.

It stands in for Slack / e-mail / PagerDuty: every notification is printed to the
terminal and appended to alertmanager/alerts.log, so the end of the alerting
pipeline is visible and can be shown as proof.

Run:  python alertmanager/alert_receiver.py      (listens on http://localhost:5001)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.getenv("ALERT_RECEIVER_PORT", "5001"))
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.log")


class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        channel = self.path.strip("/") or "default"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        for alert in payload.get("alerts", []):
            labels = alert.get("labels", {})
            notes = alert.get("annotations", {})
            lines.append(
                f"[{now}] channel={channel:<8} {alert.get('status', '?').upper():<8} "
                f"{labels.get('alertname', '?')} severity={labels.get('severity', '-')} "
                f"instance={labels.get('instance', '-')} | {notes.get('summary', '')}"
            )

        for line in lines:
            print(line, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write("\n".join(lines) + "\n")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args) -> None:  # silence the default per-request access log
        pass


if __name__ == "__main__":
    print(f"Alert receiver listening on http://localhost:{PORT}  (log: {LOG_FILE})", flush=True)
    HTTPServer(("0.0.0.0", PORT), AlertHandler).serve_forever()
