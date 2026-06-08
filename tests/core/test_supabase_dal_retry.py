"""ROB-4017 follow-up: SupabaseDal's httpx transport must retry transient
``RemoteProtocolError``s ("Server disconnected").

Supabase's edge (Cloudflare / Kong / load balancer) closes idle keep-alive
connections server-side. One ``SupabaseDal`` client is shared across the
conversation worker, realtime callbacks and request threads, so a pooled
connection the edge already closed gets reused and the next request raises
``RemoteProtocolError: Server disconnected without sending a response`` *before*
the request reaches Supabase — which makes it safe to retry on a fresh
connection. Hardening at the transport (rather than around postgrest's
``execute``) means every Supabase sub-client — postgrest, auth/gotrue, storage,
realtime — is covered uniformly. This is the hardening Supabase support
recommended (mirrors relay#573 / ROB-4012).

These tests are deterministic (no network): they drive
``SupabaseRetryTransport.handle_request`` directly with the base transport's
``handle_request`` patched to raise/return on demand.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from holmes.core.supabase_dal import SupabaseRetryTransport


def _server_disconnected() -> httpx.RemoteProtocolError:
    return httpx.RemoteProtocolError("Server disconnected without sending a response.")


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://example.supabase.co/rest/v1/Issues")


def test_transport_retries_on_remote_protocol_error_then_succeeds(monkeypatch):
    transport = SupabaseRetryTransport(disconnect_retries=3)
    response = MagicMock(name="response")
    calls = {"n": 0}

    def base_handle(_self, _request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _server_disconnected()
        return response

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", base_handle)

    assert transport.handle_request(_request()) is response
    assert calls["n"] == 2  # failed once, retried once, succeeded


def test_transport_reraises_after_exhausting_retries(monkeypatch):
    transport = SupabaseRetryTransport(disconnect_retries=3)
    calls = {"n": 0}

    def always_disconnect(_self, _request):
        calls["n"] += 1
        raise _server_disconnected()

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", always_disconnect)

    with pytest.raises(httpx.RemoteProtocolError):
        transport.handle_request(_request())
    assert calls["n"] == 3  # disconnect_retries


def test_transport_does_not_retry_other_errors(monkeypatch):
    transport = SupabaseRetryTransport(disconnect_retries=3)
    calls = {"n": 0}

    def base_handle(_self, _request):
        calls["n"] += 1
        raise httpx.ConnectTimeout("connect timed out")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", base_handle)

    with pytest.raises(httpx.ConnectTimeout):
        transport.handle_request(_request())
    assert calls["n"] == 1  # not a RemoteProtocolError -> no retry


def test_transport_retry_count_is_clamped_to_at_least_one(monkeypatch):
    # A misconfigured 0/negative retry budget must still attempt the request once.
    transport = SupabaseRetryTransport(disconnect_retries=0)
    response = MagicMock(name="response")
    calls = {"n": 0}

    def base_handle(_self, _request):
        calls["n"] += 1
        return response

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", base_handle)

    assert transport.handle_request(_request()) is response
    assert calls["n"] == 1
