"""Check available Azure OpenAI deployments."""
import httpx
from app.config import settings

endpoint = settings.azure_openai_endpoint.rstrip("/")
url = f"{endpoint}/openai/deployments?api-version=2025-01-01-preview"
r = httpx.get(url, headers={"api-key": settings.azure_openai_api_key}, timeout=10)
if r.status_code == 200:
    data = r.json()
    for d in data.get("data", []):
        print(f"  {d['id']:30s} -> {d['model']}")
else:
    print(f"Failed: {r.status_code} {r.text[:200]}")
