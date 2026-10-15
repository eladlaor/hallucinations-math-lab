"""GoogleAccessToken tests with fake credentials (no network, no ADC).

Contract pinned here:
  - A still-valid token is returned without a refresh.
  - An expired token is refreshed before being returned.
  - A refresh that yields no token fails fast.
"""

import pytest

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
