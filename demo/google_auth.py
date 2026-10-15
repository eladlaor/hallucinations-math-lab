"""Short-lived Google access tokens for calling a vLLM server on Vertex AI.

A Vertex endpoint accepts `Authorization: Bearer <access token>`, and those
tokens expire after about an hour. `GoogleAccessToken` is passed to the
OpenAI client as its `api_key`: the client calls it before every
authenticated request, and it refreshes the credentials only when the
current token is no longer valid.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from hallucinations_math_lab.constants import GOOGLE_CLOUD_PLATFORM_SCOPE, LOG_EXTRA_KEY

logger = logging.getLogger(__name__)


class RefreshableCredentials(Protocol):
    """The part of `google.auth.credentials.Credentials` this module uses."""

    @property
    def valid(self) -> bool: ...
    @property
    def token(self) -> str | None: ...
    def refresh(self, request: Any) -> None: ...


class GoogleAccessToken:
    """Callable returning a currently valid Google OAuth access token."""

    def __init__(self, credentials: RefreshableCredentials | None = None) -> None:
        """
        Args:
            credentials: Pre-built credentials (tests, or a custom identity).
                When omitted, Application Default Credentials are loaded on
                first use with the cloud-platform scope.
        """
        self._credentials = credentials
        self._lock = threading.Lock()

    def _ensure_credentials(self) -> RefreshableCredentials:
        if self._credentials is not None:
            return self._credentials
        try:
            import google.auth

            credentials, project = google.auth.default(scopes=[GOOGLE_CLOUD_PLATFORM_SCOPE])
        except Exception as e:
            raise RuntimeError(
                "Could not load Google Application Default Credentials for VLLM_AUTH=google. "
                f"Run `gcloud auth application-default login` or set "
                f"GOOGLE_APPLICATION_CREDENTIALS: {e!s}"
            ) from e
        logger.info(
            "Loaded Google credentials for vLLM requests.",
            extra={LOG_EXTRA_KEY: {"project": project}},
        )
        self._credentials = credentials
        return credentials

    def __call__(self) -> str:
        # One refresh at a time: concurrent requests would otherwise each
        # refresh the same expired credentials.
        with self._lock:
            credentials = self._ensure_credentials()
            if not credentials.valid:
                from google.auth.transport.requests import Request

                try:
                    credentials.refresh(Request())
                except Exception as e:
                    raise RuntimeError(f"Failed to refresh Google access token: {e!s}") from e
            token = credentials.token
        if not token:
            raise RuntimeError("Google credentials refreshed but returned no access token.")
        return str(token)
