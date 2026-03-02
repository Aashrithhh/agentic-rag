"""Managed secrets vault integration + automated key rotation.

Supports:
  1. Azure Key Vault (production)
  2. HashiCorp Vault (self-hosted)
  3. Environment variables (development fallback)

Secret names follow the convention: ``rag--<secret-name>``
(e.g., ``rag--openai-api-key``, ``rag--cohere-api-key``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class VaultClient:
    """Abstract vault client with caching and rotation support."""

    def __init__(self, provider: str = "env"):
        self.provider = provider
        self._cache: dict[str, tuple[str, float]] = {}  # name → (value, fetched_at)
        self._cache_ttl = 3600.0  # 1 hour cache for vault secrets
        self._lock = threading.Lock()
        self._rotation_callbacks: list[callable] = []

    def get_secret(self, name: str, default: str = "") -> str:
        """Retrieve a secret by name, with caching."""
        with self._lock:
            cached = self._cache.get(name)
            if cached and (time.monotonic() - cached[1]) < self._cache_ttl:
                return cached[0]

        value = self._fetch_secret(name, default)
        with self._lock:
            self._cache[name] = (value, time.monotonic())
        return value

    def _fetch_secret(self, name: str, default: str = "") -> str:
        """Fetch from the configured vault provider."""
        if self.provider == "azure_keyvault":
            return self._fetch_azure_keyvault(name, default)
        elif self.provider == "hashicorp":
            return self._fetch_hashicorp_vault(name, default)
        else:
            return self._fetch_env(name, default)

    def _fetch_env(self, name: str, default: str = "") -> str:
        """Fetch from environment variables (development)."""
        env_name = name.upper().replace("-", "_").replace("rag__", "")
        return os.environ.get(env_name, default)

    def _fetch_azure_keyvault(self, name: str, default: str = "") -> str:
        """Fetch from Azure Key Vault."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient

            vault_url = os.environ.get("AZURE_KEYVAULT_URL", "")
            if not vault_url:
                logger.warning("AZURE_KEYVAULT_URL not set — falling back to env")
                return self._fetch_env(name, default)

            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            secret = client.get_secret(name)
            logger.debug("Fetched secret '%s' from Azure Key Vault", name)
            return secret.value or default

        except ImportError:
            logger.warning("azure-identity not installed — falling back to env")
            return self._fetch_env(name, default)
        except Exception as exc:
            logger.error("Azure Key Vault fetch failed for '%s': %s", name, exc)
            return self._fetch_env(name, default)

    def _fetch_hashicorp_vault(self, name: str, default: str = "") -> str:
        """Fetch from HashiCorp Vault."""
        try:
            import hvac

            vault_addr = os.environ.get("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.environ.get("VAULT_TOKEN", "")

            client = hvac.Client(url=vault_addr, token=vault_token)
            secret = client.secrets.kv.v2.read_secret_version(
                path=f"rag/{name}",
                mount_point="secret",
            )
            value = secret["data"]["data"].get("value", default)
            logger.debug("Fetched secret '%s' from HashiCorp Vault", name)
            return value

        except ImportError:
            logger.warning("hvac not installed — falling back to env")
            return self._fetch_env(name, default)
        except Exception as exc:
            logger.error("HashiCorp Vault fetch failed for '%s': %s", name, exc)
            return self._fetch_env(name, default)

    def invalidate_cache(self, name: str | None = None) -> None:
        """Invalidate cached secrets (all or specific)."""
        with self._lock:
            if name:
                self._cache.pop(name, None)
            else:
                self._cache.clear()

    def rotate_secret(self, name: str, new_value: str) -> bool:
        """Rotate a secret — store new value and invalidate cache."""
        try:
            if self.provider == "azure_keyvault":
                return self._rotate_azure(name, new_value)
            elif self.provider == "hashicorp":
                return self._rotate_hashicorp(name, new_value)
            else:
                logger.warning("Secret rotation not supported for provider '%s'", self.provider)
                return False
        except Exception as exc:
            logger.error("Secret rotation failed for '%s': %s", name, exc)
            return False
        finally:
            self.invalidate_cache(name)

    def _rotate_azure(self, name: str, new_value: str) -> bool:
        """Rotate a secret in Azure Key Vault."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient

            vault_url = os.environ.get("AZURE_KEYVAULT_URL", "")
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            client.set_secret(name, new_value)
            logger.info("Rotated secret '%s' in Azure Key Vault", name)
            return True
        except Exception as exc:
            logger.error("Azure Key Vault rotation failed: %s", exc)
            return False

    def _rotate_hashicorp(self, name: str, new_value: str) -> bool:
        """Rotate a secret in HashiCorp Vault."""
        try:
            import hvac

            vault_addr = os.environ.get("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.environ.get("VAULT_TOKEN", "")
            client = hvac.Client(url=vault_addr, token=vault_token)
            client.secrets.kv.v2.create_or_update_secret(
                path=f"rag/{name}",
                secret={"value": new_value},
                mount_point="secret",
            )
            logger.info("Rotated secret '%s' in HashiCorp Vault", name)
            return True
        except Exception as exc:
            logger.error("HashiCorp Vault rotation failed: %s", exc)
            return False


# ── Key rotation scheduler ───────────────────────────────────────────


class KeyRotationScheduler:
    """Automated key rotation on a configurable schedule."""

    def __init__(
        self,
        vault: VaultClient,
        rotation_interval_hours: int = 720,  # 30 days default
    ):
        self.vault = vault
        self.interval = rotation_interval_hours * 3600
        self._last_rotation: dict[str, float] = {}
        self._running = False

    def check_rotation_needed(self, secret_name: str) -> bool:
        """Check if a secret is due for rotation."""
        last = self._last_rotation.get(secret_name, 0)
        return (time.time() - last) > self.interval

    def start_background_rotation(self, secret_names: list[str]) -> None:
        """Start background thread to check and rotate secrets."""
        def _rotation_loop():
            self._running = True
            while self._running:
                for name in secret_names:
                    if self.check_rotation_needed(name):
                        logger.info("Secret '%s' is due for rotation check", name)
                        # In production, this would generate a new key and
                        # update the upstream provider (e.g., regenerate API key)
                        self._last_rotation[name] = time.time()
                time.sleep(3600)  # Check hourly

        thread = threading.Thread(
            target=_rotation_loop, daemon=True, name="key-rotation"
        )
        thread.start()
        logger.info("Key rotation scheduler started for %d secrets", len(secret_names))

    def stop(self) -> None:
        self._running = False


# Module-level vault singleton
vault = VaultClient(provider=os.environ.get("VAULT_PROVIDER", "env"))
