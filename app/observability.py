"""Production observability — Prometheus metrics export + alerting rules.

Bridges the in-process ``app.metrics`` registry to Prometheus format
and defines alerting thresholds for Grafana/Alertmanager.

Components:
  1. Prometheus exporter — /metrics/prometheus endpoint
  2. Alert definitions — encoded as rules for Prometheus Alertmanager
  3. Grafana dashboard JSON generator
  4. Log export adapter for Datadog/ELK
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── Prometheus text format exporter ──────────────────────────────────


def metrics_to_prometheus(snapshot: dict[str, Any]) -> str:
    """Convert internal metrics snapshot to Prometheus text exposition format.

    Produces HELP, TYPE, and metric lines per Prometheus conventions.
    """
    lines: list[str] = []

    counters = snapshot.get("counters", {})
    histograms = snapshot.get("histograms", {})

    # Counters
    for name, value in sorted(counters.items()):
        prom_name = _sanitize_name(name)
        lines.append(f"# HELP {prom_name} Counter: {name}")
        lines.append(f"# TYPE {prom_name} counter")
        lines.append(f"{prom_name} {value}")

    # Histograms — emit as gauges with quantile labels
    for name, stats in sorted(histograms.items()):
        prom_name = _sanitize_name(name)
        lines.append(f"# HELP {prom_name} Histogram: {name}")
        lines.append(f"# TYPE {prom_name} summary")
        count = stats.get("count", 0)
        if count > 0:
            lines.append(f'{prom_name}{{quantile="0.5"}} {stats.get("p50", 0)}')
            lines.append(f'{prom_name}{{quantile="0.95"}} {stats.get("p95", 0)}')
            lines.append(f'{prom_name}{{quantile="0.99"}} {stats.get("p99", 0)}')
            lines.append(f"{prom_name}_count {count}")
            lines.append(f"{prom_name}_sum {stats.get('mean', 0) * count}")

    # Uptime
    lines.append("# HELP rag_up Whether the RAG service is up")
    lines.append("# TYPE rag_up gauge")
    lines.append("rag_up 1")

    return "\n".join(lines) + "\n"


def _sanitize_name(name: str) -> str:
    """Convert metric name to Prometheus-safe format."""
    return "rag_" + name.replace(".", "_").replace("-", "_").replace("/", "_")


# ── Alert definitions ────────────────────────────────────────────────


ALERT_RULES: list[dict[str, Any]] = [
    {
        "name": "HighRetryRate",
        "expr": "rate(rag_llm_call_fallback_total[5m]) > 0.1",
        "for": "5m",
        "severity": "warning",
        "description": "LLM fallback rate exceeds 10% over 5 minutes",
        "runbook": "Check OpenAI API status, review error logs",
    },
    {
        "name": "CacheMissSpike",
        "expr": "(rag_cache_misses / (rag_cache_hits + rag_cache_misses)) > 0.8",
        "for": "10m",
        "severity": "warning",
        "description": "Cache miss rate exceeds 80% for 10 minutes",
        "runbook": "Check if cache was recently cleared or if query patterns changed",
    },
    {
        "name": "ValidationFailSpike",
        "expr": "rate(rag_node_invocations_validator[5m]) > 0 and "
                "(rate(rag_validation_fail_total[5m]) / rate(rag_node_invocations_validator[5m])) > 0.3",
        "for": "5m",
        "severity": "critical",
        "description": "Validation failure rate exceeds 30%",
        "runbook": "Check external validation API health and claim extraction quality",
    },
    {
        "name": "HighP95Latency",
        "expr": "rag_http_request_seconds_query{quantile=\"0.95\"} > 30",
        "for": "5m",
        "severity": "critical",
        "description": "P95 query latency exceeds 30 seconds (SLO breach)",
        "runbook": "Check LLM API latency, database performance, cache hit rates",
    },
    {
        "name": "DBSlowQueries",
        "expr": "rag_node_latency_retriever{quantile=\"0.95\"} > 5",
        "for": "5m",
        "severity": "warning",
        "description": "Retriever P95 latency exceeds 5 seconds",
        "runbook": "Check PostgreSQL performance, HNSW index health, connection pool",
    },
    {
        "name": "HighErrorRate",
        "expr": "rate(rag_llm_call_failure_total[5m]) / rate(rag_llm_call_success_total[5m]) > 0.05",
        "for": "5m",
        "severity": "critical",
        "description": "LLM call error rate exceeds 5%",
        "runbook": "Check API keys, rate limits, model availability",
    },
    {
        "name": "CircuitBreakerOpen",
        "expr": "rag_circuit_breaker_state == 2",
        "for": "1m",
        "severity": "critical",
        "description": "Circuit breaker is open for an upstream service",
        "runbook": "Check upstream service health, review recent failures",
    },
    {
        "name": "PIIDetected",
        "expr": "rate(rag_pii_detections_total[5m]) > 1",
        "for": "5m",
        "severity": "warning",
        "description": "PII detection rate is elevated",
        "runbook": "Review PII redaction pipeline, check incoming documents",
    },
]


def generate_prometheus_rules() -> str:
    """Generate Prometheus alerting rules YAML."""
    rules = []
    for alert in ALERT_RULES:
        rules.append(
            f"  - alert: {alert['name']}\n"
            f"    expr: {alert['expr']}\n"
            f"    for: {alert['for']}\n"
            f"    labels:\n"
            f"      severity: {alert['severity']}\n"
            f"    annotations:\n"
            f"      description: \"{alert['description']}\"\n"
            f"      runbook: \"{alert['runbook']}\"\n"
        )

    return (
        "groups:\n"
        "  - name: rag_alerts\n"
        "    rules:\n" + "\n".join(rules)
    )


# ── Grafana dashboard generator ──────────────────────────────────────


def generate_grafana_dashboard() -> dict[str, Any]:
    """Generate a Grafana dashboard JSON for the RAG pipeline."""
    return {
        "dashboard": {
            "title": "Agentic RAG Pipeline",
            "uid": "rag-pipeline-main",
            "timezone": "utc",
            "refresh": "30s",
            "panels": [
                {
                    "title": "Query Latency (P95)",
                    "type": "timeseries",
                    "targets": [
                        {"expr": 'rag_http_request_seconds_query{quantile="0.95"}',
                         "legendFormat": "P95 Latency"},
                        {"expr": 'rag_http_request_seconds_query{quantile="0.5"}',
                         "legendFormat": "P50 Latency"},
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "thresholds": [
                        {"value": 30, "colorMode": "critical"},
                        {"value": 15, "colorMode": "warning"},
                    ],
                },
                {
                    "title": "Error Rate",
                    "type": "stat",
                    "targets": [
                        {"expr": "rate(rag_llm_call_failure_total[5m]) / "
                                 "rate(rag_llm_call_success_total[5m])",
                         "legendFormat": "Error Rate"},
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
                },
                {
                    "title": "Cache Hit Rate",
                    "type": "gauge",
                    "targets": [
                        {"expr": "rag_cache_hits / (rag_cache_hits + rag_cache_misses)",
                         "legendFormat": "Hit Rate"},
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
                },
                {
                    "title": "Node Latency Breakdown",
                    "type": "timeseries",
                    "targets": [
                        {"expr": 'rag_node_latency_retriever{quantile="0.95"}',
                         "legendFormat": "Retriever"},
                        {"expr": 'rag_node_latency_grader{quantile="0.95"}',
                         "legendFormat": "Grader"},
                        {"expr": 'rag_node_latency_generator{quantile="0.95"}',
                         "legendFormat": "Generator"},
                        {"expr": 'rag_node_latency_validator{quantile="0.95"}',
                         "legendFormat": "Validator"},
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                },
                {
                    "title": "Circuit Breaker Status",
                    "type": "stat",
                    "targets": [
                        {"expr": "rag_circuit_breaker_state",
                         "legendFormat": "{{service}}"},
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 12, "y": 4},
                },
                {
                    "title": "Rerank & Intent Stats",
                    "type": "timeseries",
                    "targets": [
                        {"expr": "rate(rag_rerank_invocations[5m])",
                         "legendFormat": "Rerank/s"},
                        {"expr": "rate(rag_intent_classifications[5m])",
                         "legendFormat": "Intent Classify/s"},
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 18, "y": 4},
                },
                {
                    "title": "PII Detections",
                    "type": "timeseries",
                    "targets": [
                        {"expr": "rate(rag_pii_detections_total[5m])",
                         "legendFormat": "PII Detections/min"},
                    ],
                    "gridPos": {"h": 4, "w": 12, "x": 12, "y": 8},
                },
                {
                    "title": "HITL Review Queue",
                    "type": "stat",
                    "targets": [
                        {"expr": "rag_hitl_pending_reviews",
                         "legendFormat": "Pending Reviews"},
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 12, "y": 12},
                },
            ],
        }
    }


# ── Structured log exporter ─────────────────────────────────────────


class MetricsExporter:
    """Periodically export metrics to external systems."""

    def __init__(self, export_interval_seconds: int = 30):
        self._interval = export_interval_seconds
        self._running = False

    def export_once(self) -> dict[str, Any]:
        """Export current metrics snapshot."""
        from app.metrics import metrics
        from app.cache import cache

        snapshot = metrics.snapshot()
        cache_stats = cache.stats()

        export = {
            "timestamp": time.time(),
            "pipeline_metrics": snapshot,
            "cache_stats": cache_stats,
        }

        # Try SLO check
        try:
            from app.slo import full_slo_check
            slo_report = full_slo_check(snapshot)
            export["slo_status"] = slo_report.to_dict()
        except Exception:
            pass

        return export

    def start_background_export(self) -> None:
        """Start periodic background export (call from startup)."""
        import threading

        def _export_loop():
            self._running = True
            while self._running:
                try:
                    data = self.export_once()
                    logger.debug("Metrics exported: %d counters, %d histograms",
                                 len(data.get("pipeline_metrics", {}).get("counters", {})),
                                 len(data.get("pipeline_metrics", {}).get("histograms", {})))
                except Exception as exc:
                    logger.error("Metrics export failed: %s", exc)
                time.sleep(self._interval)

        thread = threading.Thread(target=_export_loop, daemon=True, name="metrics-exporter")
        thread.start()
        logger.info("Background metrics exporter started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False


# Module-level exporter singleton
exporter = MetricsExporter()
