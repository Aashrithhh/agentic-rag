"""Load test for the Agentic RAG API.

Run:
    locust -f tests/load/locustfile.py --host http://localhost:8080 \
           --headless -u 10 -r 2 -t 60s
"""

from __future__ import annotations

import os

from locust import HttpUser, between, task


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {}
        key = os.environ.get("API_KEY_FOR_LOAD_TEST", "")
        if key:
            self.headers["X-API-Key"] = key

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(3)
    def list_cases(self):
        self.client.get("/api/v1/cases", headers=self.headers)

    @task(5)
    def query_big_thorium(self):
        self.client.post("/api/v1/query", json={
            "query": "What safety incidents were reported at Big Thorium?",
            "case_id": "big-thorium",
        }, headers=self.headers)
