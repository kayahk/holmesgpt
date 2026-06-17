"""ROB-4017: pin ``SupabaseRetryTransport``'s retry contract.

Deterministically drive ``SupabaseRetryTransport.handle_request`` (with the base
transport's ``handle_request`` patched to raise/return on demand) to verify it
retries ``RemoteProtocolError`` on a fresh connection, stops after
``disconnect_retries``, reraises the original exception, and does not retry other
errors. See ``SupabaseRetryTransport`` for why this retry is needed and safe.
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
