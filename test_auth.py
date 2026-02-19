"""Quick test of Azure OpenAI keys."""
import httpx
import os
from pathlib import Path
from dotenv import dotenv_values

# System env values
sys_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
sys_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
sys_deploy = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# .env file values
dotenv = dotenv_values(Path(__file__).parent / ".env")
env_key = dotenv.get("AZURE_OPENAI_API_KEY", "")
env_endpoint = dotenv.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
env_deploy = dotenv.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

def test_key(label, endpoint, deploy, key):
    url = f"{endpoint}/openai/deployments/{deploy}/chat/completions?api-version=2025-01-01-preview"
    print(f"\n[{label}] {endpoint} / {deploy}")
    try:
        r = httpx.post(
            url,
            headers={"api-key": key},
            json={"messages": [{"role": "user", "content": "say hi"}], "max_tokens": 5},
            timeout=15,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  Response: {r.json()['choices'][0]['message']['content']}")
        else:
            print(f"  Error: {r.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")

test_key("SYSTEM ENV", sys_endpoint, sys_deploy, sys_key)
test_key(".ENV FILE", env_endpoint, env_deploy, env_key)
