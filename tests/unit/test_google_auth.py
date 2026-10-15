"""GoogleAccessToken tests with fake credentials (no network, no ADC).

Contract pinned here:
  - A still-valid token is returned without a refresh.
  - An expired token is refreshed before being returned.
  - A refresh that yields no token fails fast.
  - The real OpenAI client calls the provider on chat and embedding requests.
"""

import httpx2 as httpx
import pytest
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

from demo.google_auth import GoogleAccessToken


class FakeCredentials:
    def __init__(self, valid: bool, token: str | None, refreshed_token: str | None) -> None:
        self.valid = valid
        self.token = token
        self._refreshed_token = refreshed_token
        self.refresh_calls = 0

    def refresh(self, request: object) -> None:
        self.refresh_calls += 1
        self.token = self._refreshed_token
        self.valid = self._refreshed_token is not None


def test_valid_token_is_reused():
    creds = FakeCredentials(valid=True, token="tok-1", refreshed_token="tok-2")
    assert GoogleAccessToken(credentials=creds)() == "tok-1"
    assert creds.refresh_calls == 0


def test_expired_token_is_refreshed():
    creds = FakeCredentials(valid=False, token="stale", refreshed_token="fresh")
    assert GoogleAccessToken(credentials=creds)() == "fresh"
    assert creds.refresh_calls == 1


def test_refresh_without_token_fails_fast():
    creds = FakeCredentials(valid=False, token=None, refreshed_token=None)
    with pytest.raises(RuntimeError, match="no access token"):
        GoogleAccessToken(credentials=creds)()


def test_openai_client_sends_a_refreshed_token_on_chat_and_embedding_requests():
    """Contract with the real OpenAI SDK (no mocks of it): the token provider
    is consulted per request, including the hand-built /embeddings call.
    (openai 3 ships its HTTP layer as `httpx2`, hence the import alias.)"""
    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers["Authorization"])
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "m",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.0]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    creds = FakeCredentials(valid=False, token=None, refreshed_token="fresh-token")
    client = OpenAI(
        base_url="https://vertex.example/v1",
        api_key=GoogleAccessToken(credentials=creds),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    client.post(
        "/embeddings",
        body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        cast_to=CreateEmbeddingResponse,
        options={"security": {"bearer_auth": True}},
    )

    assert seen_auth == ["Bearer fresh-token", "Bearer fresh-token"]
    assert creds.refresh_calls == 1
