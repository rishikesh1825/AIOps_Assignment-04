# AI Disclosure

**Tool used:** Claude (Anthropic), through the claude.ai chat interface.

## How AI was used

- **Dashboard:** AI generated the Grafana dashboard JSON (`dashboard.json`) from my requirements. I reviewed and tested it on my machine.
- **Debugging:** AI helped diagnose and fix errors I hit while running the stack: a port conflict
  for the metrics endpoint, low disk space, Grafana in Docker being unable to reach Prometheus,
  Alertmanager not starting, the Grafana Prometheus plugin going missing, and the dashboard
  exporting in Grafana's v2 format.
- **Documentation:** AI generated the first draft of `README.md`.

## PromQL and regex attribution

As required by the Code of Conduct, every AI-generated PromQL expression and regex in
`prometheus/alert_rules.yml`, `prometheus/prometheus.yml` and `alertmanager/alertmanager.yml` is
commented with the prompt used. The dashboard's PromQL was generated from the request:

> "Build a 16-panel Grafana dashboard for these 20 metrics plus node_exporter, flowing from health to
> system resources, following the seven plotting commandments, with no dual axes."

Each dashboard panel's description explains what its query computes.

## Representative prompts

1. "The caption was generated, but what is the warning about?" (Ubuntu low-disk-space warning and
   the metrics server reporting its port as busy)
2. "Unable to connect to localhost:9093 ??"
3. "what is happening?" (every Grafana dashboard panel showing errors)
4. "There is no option in the dropdown?" (Grafana's data source selector was empty)

## Verification

Configs were validated with `promtool check config` and `amtool check-config`; metrics were checked
at `/metrics`, targets and alerts in the Prometheus UI, and every dashboard panel against live data.
