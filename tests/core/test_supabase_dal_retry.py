"""ROB-4017 follow-up: SupabaseDal must retry postgrest queries that fail with a
transient ``httpx.RemoteProtocolError`` ("Server disconnected").

Supabase's edge (Cloudflare / Kong / load balancer) closes idle keep-alive
connections server-side. The DAL client is shared across the conversation
worker, realtime callbacks and request threads, so a pooled connection the edge
has already closed gets reused and the next query raises
``RemoteProtocolError: Server disconnected without sending a response`` *before*
the request reaches Supabase — which makes it safe to retry. This is the
hardening Supabase support recommended for these errors (mirrors relay#573 /
ROB-4012); disabling HTTP/2 reduces but does not eliminate them.

These tests are deterministic (no network): they patch the postgrest
``SyncQueryRequestBuilder.execute`` that ``SupabaseDal`` wraps and assert the
retry behaviour, with tenacity's backoff neutralised so nothing actually sleeps.
"""

from unittest.mock import MagicMock

import httpx
import pytest
from postgrest._sync.request_builder import SyncQueryRequestBuilder
from postgrest.exceptions import APIError as PGAPIError
from tenacity import wait_fixed

import holmes.core.supabase_dal as supabase_dal_module
from holmes.core.supabase_dal import SupabaseDal


@pytest.fixture
def no_backoff(monkeypatch):
    """Neutralise tenacity's wait so the retry tests don't sleep.

    ``patch_postgrest_execute`` reads ``wait_exponential`` from the module
    namespace when it builds the decorator, so patching it here (before the DAL
    is patched) makes the ``Retrying`` controller wait zero seconds.
    """
    monkeypatch.setattr(
        supabase_dal_module, "wait_exponential", lambda *a, **k: wait_fixed(0)
    )


def _make_dal() -> SupabaseDal:
    # Bypass __init__ (no network / token needed). patch_postgrest_execute only
    # touches self._original_execute, self.sign_in and self.client.
    return SupabaseDal.__new__(SupabaseDal)


def _remote_protocol_error() -> httpx.RemoteProtocolError:
    return httpx.RemoteProtocolError("Server disconnected without sending a response.")


def test_execute_retries_on_remote_protocol_error_then_succeeds(monkeypatch, no_backoff):
    calls = {"n": 0}

    def flaky(_self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _remote_protocol_error()
        return "ok"

    monkeypatch.setattr(SyncQueryRequestBuilder, "execute", flaky, raising=True)
    dal = _make_dal()
    dal.patch_postgrest_execute()

    result = SyncQueryRequestBuilder.execute(object())

    assert result == "ok"
    assert calls["n"] == 2  # failed once, retried once, succeeded


def test_execute_reraises_remote_protocol_error_after_exhausting_retries(
    monkeypatch, no_backoff
):
    calls = {"n": 0}

    def always_disconnect(_self):
        calls["n"] += 1
        raise _remote_protocol_error()

    monkeypatch.setattr(
        SyncQueryRequestBuilder, "execute", always_disconnect, raising=True
    )
    dal = _make_dal()
    dal.patch_postgrest_execute()

    with pytest.raises(httpx.RemoteProtocolError):
        SyncQueryRequestBuilder.execute(object())

    assert calls["n"] == 3  # stop_after_attempt(3)


def test_execute_does_not_retry_non_transport_errors(monkeypatch, no_backoff):
    calls = {"n": 0}

    def boom(_self):
        calls["n"] += 1
        raise ValueError("not a transport error")

    monkeypatch.setattr(SyncQueryRequestBuilder, "execute", boom, raising=True)
    dal = _make_dal()
    dal.patch_postgrest_execute()

    with pytest.raises(ValueError):
        SyncQueryRequestBuilder.execute(object())

    assert calls["n"] == 1  # not retried


def test_expired_jwt_signs_in_again_and_retries(monkeypatch, no_backoff):
    calls = {"n": 0}

    def expired_then_ok(_self):
        calls["n"] += 1
        if calls["n"] == 1:
            err = PGAPIError.__new__(PGAPIError)
            err.message = "JWT expired"
            err.code = "PGRST301"
            raise err
        return "ok"

    monkeypatch.setattr(
        SyncQueryRequestBuilder, "execute", expired_then_ok, raising=True
    )
    dal = _make_dal()
    dal.sign_in = MagicMock()
    dal.client = MagicMock()

    class _Session:
        session = None

    dal.patch_postgrest_execute()
    result = SyncQueryRequestBuilder.execute(_Session())

    assert result == "ok"
    dal.sign_in.assert_called_once()
    assert calls["n"] == 2


def test_expired_jwt_path_also_retries_transport_errors(monkeypatch, no_backoff):
    # After a JWT re-sign-in, the retried query should itself survive a transient
    # RemoteProtocolError (the re-execute goes through the same transport retry).
    calls = {"n": 0}

    def sequence(_self):
        calls["n"] += 1
        if calls["n"] == 1:
            err = PGAPIError.__new__(PGAPIError)
            err.message = "JWT expired"
            err.code = "PGRST301"
            raise err
        if calls["n"] == 2:
            raise _remote_protocol_error()
        return "ok"

    monkeypatch.setattr(SyncQueryRequestBuilder, "execute", sequence, raising=True)
    dal = _make_dal()
    dal.sign_in = MagicMock()
    dal.client = MagicMock()

    class _Session:
        session = None

    dal.patch_postgrest_execute()
    result = SyncQueryRequestBuilder.execute(_Session())

    assert result == "ok"
    dal.sign_in.assert_called_once()
    assert calls["n"] == 3  # expired -> re-sign-in -> disconnect -> retry -> ok
