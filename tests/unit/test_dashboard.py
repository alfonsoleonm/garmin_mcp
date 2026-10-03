"""Unit tests for GET /api/dashboard."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import garmin_mcp.dashboard as dash
from garmin_mcp.http_auth import PathSecretMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(method="GET", headers=None, path="/api/dashboard", query_string=""):
    """Build a minimal Starlette Request from plain dicts."""
    from starlette.requests import Request

    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    qs = query_string.encode() if isinstance(query_string, str) else query_string
    scope = {
        "type": "http",
        "method": method.upper(),
        "path": path,
        "query_string": qs,
        "headers": raw_headers,
    }
    return Request(scope)


def _sync_run_blocking(fn, *args):
    """Replacement for _run_blocking that calls fns synchronously in tests."""
    return fn(*args)


@pytest.fixture(autouse=True)
def reset_dashboard(monkeypatch):
    """Isolate module globals between tests."""
    dash._cache.clear()
    # Replace the thread-pool wrapper with a synchronous caller so mock
    # return values are returned directly without spawning real threads.
    monkeypatch.setattr(dash, "_run_blocking", AsyncMock(side_effect=_sync_run_blocking))
    yield
    dash._cache.clear()


@pytest.fixture
def client():
    """A MagicMock wired as the Garmin client for dashboard calls."""
    mock = MagicMock()
    # Default happy-path return values
    mock.get_sleep_data.return_value = {
        "dailySleepDTO": {
            "sleepTimeSeconds": 25200,
            "sleepScores": {"overall": {"value": 78, "qualifierKey": "GOOD"}},
            "deepSleepSeconds": 4000,
            "lightSleepSeconds": 12000,
            "remSleepSeconds": 6000,
            "awakeSleepSeconds": 3200,
        }
    }
    mock.get_hrv_data.return_value = {
        "hrvSummary": {
            "lastNightAvg": 54,
            "lastNight5MinHigh": 72,
            "weeklyAvg": 52,
            "status": "BALANCED",
            "feedbackPhrase": "HRV_BALANCED_3",
            "baseline": {"balancedLow": 45, "balancedUpper": 65},
        }
    }
    mock.get_stats.return_value = {
        "bodyBatteryMostRecentValue": 72,
        "bodyBatteryHighestValue": 95,
        "bodyBatteryLowestValue": 18,
        "bodyBatteryChargedValue": 77,
        "bodyBatteryDrainedValue": 23,
        "restingHeartRate": 52,
        "lastSevenDaysAvgRestingHeartRate": 53,
    }
    mock.get_training_readiness.return_value = [
        {
            "score": 68,
            "level": "MODERATE",
            "feedbackShort": "TRAINING_READINESS_MODERATE",
            "sleepScore": 75,
            "hrvWeeklyAverage": 52,
            "acuteLoad": 245,
        }
    ]
    mock.get_training_status.return_value = {
        "mostRecentTrainingStatus": {
            "latestTrainingStatusData": {
                "device-1": {
                    "trainingStatus": "PRODUCTIVE",
                    "trainingStatusFeedbackPhrase": "TRAINING_STATUS_PRODUCTIVE",
                    "acuteTrainingLoadDTO": {
                        "dailyTrainingLoadAcute": 245.3,
                        "dailyTrainingLoadChronic": 210.5,
                        "dailyAcuteChronicWorkloadRatio": 1.17,
                        "acwrStatus": "OPTIMAL",
                    },
                }
            }
        }
    }
    dash.configure(mock)
    return mock


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

class TestAuth:
    async def test_401_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
        req = make_request(headers={"Authorization": "Bearer anykey"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 401

    async def test_401_when_wrong_key(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "correct-key")
        req = make_request(headers={"Authorization": "Bearer wrong-key"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 401

    async def test_401_when_missing_header(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "correct-key")
        req = make_request()  # no Authorization header
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 401

    async def test_401_body_is_generic(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "correct-key")
        req = make_request(headers={"Authorization": "Bearer bad"})
        resp = await dash._dashboard_handler(req)
        import json
        body = json.loads(resp.body)
        assert "error" in body
        # Must not leak key or any health data
        assert "correct-key" not in str(body)

    async def test_401_when_malformed_scheme(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "correct-key")
        req = make_request(headers={"Authorization": "Token correct-key"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestSuccess:
    async def test_200_all_sections_present(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 200

        import json
        body = json.loads(resp.body)
        assert "date" in body
        assert "sleep" in body
        assert "hrv" in body
        assert "body_battery" in body
        assert "resting_hr" in body
        assert "training_readiness" in body
        assert "training_load" in body
        assert "errors" not in body

    async def test_hrv_fields_extracted(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        import json
        hrv = json.loads(resp.body)["hrv"]
        assert hrv["last_night_avg_ms"] == 54
        assert hrv["weekly_avg_ms"] == 52
        assert hrv["status"] == "BALANCED"

    async def test_body_battery_fields_extracted(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        import json
        bb = json.loads(resp.body)["body_battery"]
        assert bb["current"] == 72
        assert bb["highest"] == 95
        assert bb["lowest"] == 18

    async def test_sleep_has_today_and_trend(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        import json
        sleep = json.loads(resp.body)["sleep"]
        assert "today" in sleep
        assert "trend_7d" in sleep
        assert isinstance(sleep["trend_7d"], list)


# ---------------------------------------------------------------------------
# Partial failure (resilience)
# ---------------------------------------------------------------------------

class TestPartialFailure:
    async def test_failed_section_is_null_with_error_key(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        client.get_hrv_data.side_effect = RuntimeError("garmin timeout")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 200

        import json
        body = json.loads(resp.body)
        assert body["hrv"] is None
        assert "errors" in body
        assert "hrv" in body["errors"]
        assert body["errors"]["hrv"] == "RuntimeError"

    async def test_failed_section_does_not_cancel_others(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        client.get_hrv_data.side_effect = ConnectionError("network")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)

        import json
        body = json.loads(resp.body)
        # The other sections must still be populated
        assert body["sleep"] is not None
        assert body["body_battery"] is not None
        assert body["training_readiness"] is not None
        assert body["training_load"] is not None

    async def test_error_message_does_not_contain_exception_text(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        client.get_training_status.side_effect = ValueError("secret-token-abc123")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)

        import json
        body = json.loads(resp.body)
        body_str = str(body)
        assert "secret-token-abc123" not in body_str
        # Only the class name is recorded
        assert body["errors"]["training_load"] == "ValueError"

    async def test_all_sections_can_fail_returns_200(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        err = Exception("boom")
        client.get_sleep_data.side_effect = err
        client.get_hrv_data.side_effect = err
        client.get_stats.side_effect = err
        client.get_training_readiness.side_effect = err
        client.get_training_status.side_effect = err
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 200

        import json
        body = json.loads(resp.body)
        assert "date" in body
        assert len(body["errors"]) >= 4


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    async def test_second_call_uses_cache(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        await dash._dashboard_handler(req)
        await dash._dashboard_handler(req)
        # get_stats is called for body_battery + resting_hr per request.
        # Only one real set of calls should have happened (first request).
        assert client.get_hrv_data.call_count == 1

    async def test_expired_cache_refetches(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        await dash._dashboard_handler(req)
        import datetime as _dt
        today = _dt.date.today().isoformat()
        # Back-date the cache entry so it appears expired
        payload, _ = dash._cache[today]
        import time
        dash._cache[today] = (payload, time.monotonic() - dash._CACHE_TTL - 1)
        await dash._dashboard_handler(req)
        assert client.get_hrv_data.call_count == 2


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCors:
    async def test_preflight_returns_204_without_auth(self, monkeypatch, client):
        monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
        req = make_request(
            method="OPTIONS",
            headers={"Origin": "https://alfonsoleonm.github.io"},
        )
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 204

    async def test_preflight_includes_cors_headers(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(
            method="OPTIONS",
            headers={"Origin": "https://alfonsoleonm.github.io"},
        )
        resp = await dash._dashboard_handler(req)
        assert resp.headers["access-control-allow-origin"] == "https://alfonsoleonm.github.io"
        assert "Authorization" in resp.headers["access-control-allow-headers"]
        assert "Vary" in resp.headers

    async def test_allowed_origin_echoed_on_get(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(
            headers={
                "Authorization": "Bearer mykey",
                "Origin": "http://localhost:5173",
            }
        )
        resp = await dash._dashboard_handler(req)
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    async def test_unknown_origin_gets_no_cors_header(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(
            headers={
                "Authorization": "Bearer mykey",
                "Origin": "https://evil.example.com",
            }
        )
        resp = await dash._dashboard_handler(req)
        assert "access-control-allow-origin" not in resp.headers

    async def test_wildcard_never_returned(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        for origin in ["https://alfonsoleonm.github.io", "http://localhost:5173", ""]:
            req = make_request(
                headers={"Authorization": "Bearer mykey", "Origin": origin}
            )
            resp = await dash._dashboard_handler(req)
            acao = resp.headers.get("access-control-allow-origin", "")
            assert acao != "*", f"Wildcard CORS returned for origin {origin!r}"

    async def test_401_on_get_still_includes_cors_headers(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(
            headers={
                "Authorization": "Bearer wrong",
                "Origin": "https://alfonsoleonm.github.io",
            }
        )
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 401
        assert resp.headers.get("access-control-allow-origin") == "https://alfonsoleonm.github.io"


# ---------------------------------------------------------------------------
# PathSecretMiddleware: /mcp still protected, /api bypasses
# ---------------------------------------------------------------------------

class TestPathSecretMiddleware:
    def _make_middleware(self, secret="s3cret"):
        """Return middleware wrapping a recorder that logs reached paths."""
        reached = []

        async def inner(scope, receive, send):
            reached.append(scope["path"])
            from starlette.responses import PlainTextResponse
            await PlainTextResponse("ok")(scope, receive, send)

        mw = PathSecretMiddleware(inner, secret)
        return mw, reached

    async def _call(self, mw, path, method="GET"):
        from starlette.responses import Response

        responses = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            responses.append(message)

        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "query_string": b"",
            "headers": [],
        }
        await mw(scope, receive, send)
        status = next(
            (m["status"] for m in responses if m.get("type") == "http.response.start"),
            None,
        )
        return status

    async def test_mcp_without_secret_returns_404(self):
        mw, reached = self._make_middleware("s3cret")
        status = await self._call(mw, "/mcp")
        assert status == 404
        assert "/mcp" not in reached

    async def test_mcp_with_wrong_secret_returns_404(self):
        mw, reached = self._make_middleware("s3cret")
        status = await self._call(mw, "/wrong/mcp")
        assert status == 404

    async def test_mcp_with_correct_secret_passes_through(self):
        mw, reached = self._make_middleware("s3cret")
        status = await self._call(mw, "/s3cret/mcp")
        assert status == 200
        assert "/mcp" in reached

    async def test_api_dashboard_bypasses_path_secret(self):
        mw, reached = self._make_middleware("s3cret")
        status = await self._call(mw, "/api/dashboard")
        assert status == 200
        assert "/api/dashboard" in reached

    async def test_api_mcp_does_not_reach_mcp_endpoint(self):
        """An attacker cannot use the /api/ bypass to reach /mcp."""
        mw, reached = self._make_middleware("s3cret")
        # /api/mcp passes through the middleware as /api/mcp — not /mcp.
        # The inner app would route /api/mcp to its own 404; the test just
        # confirms the path arrives unchanged (no stripping), which is safe.
        await self._call(mw, "/api/mcp")
        assert "/mcp" not in reached
        assert "/api/mcp" in reached

    async def test_healthz_bypasses_path_secret(self):
        mw, reached = self._make_middleware("s3cret")
        status = await self._call(mw, "/healthz")
        assert status == 200
        assert "/healthz" in reached


# ---------------------------------------------------------------------------
# Timezone and ?date= param
# ---------------------------------------------------------------------------

class TestTimezone:
    def test_server_today_returns_a_date(self, monkeypatch):
        import datetime as _dt
        monkeypatch.delenv("DASHBOARD_TZ", raising=False)
        result = dash._server_today()
        assert isinstance(result, _dt.date)

    def test_server_today_respects_dashboard_tz(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_TZ", "America/New_York")
        result = dash._server_today()
        import datetime as _dt
        assert isinstance(result, _dt.date)

    def test_server_today_falls_back_on_invalid_tz(self, monkeypatch):
        """An unknown timezone must not crash — falls back to Mexico City."""
        monkeypatch.setenv("DASHBOARD_TZ", "Not/A/Real/Timezone")
        result = dash._server_today()
        import datetime as _dt
        assert isinstance(result, _dt.date)

    async def test_date_param_reflected_in_response(self, monkeypatch, client):
        import datetime as _dt
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        monkeypatch.setattr(dash, "_server_today", lambda: _dt.date(2026, 10, 2))
        req = make_request(
            headers={"Authorization": "Bearer mykey"},
            query_string="date=2026-10-01",
        )
        resp = await dash._dashboard_handler(req)
        import json
        assert json.loads(resp.body)["date"] == "2026-10-01"

    async def test_date_param_absent_uses_server_today(self, monkeypatch, client):
        import datetime as _dt
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        monkeypatch.setattr(dash, "_server_today", lambda: _dt.date(2026, 10, 2))
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        import json
        assert json.loads(resp.body)["date"] == "2026-10-02"

    async def test_date_param_one_day_ahead_allowed(self, monkeypatch, client):
        import datetime as _dt
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        monkeypatch.setattr(dash, "_server_today", lambda: _dt.date(2026, 10, 2))
        req = make_request(
            headers={"Authorization": "Bearer mykey"},
            query_string="date=2026-10-03",
        )
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 200

    async def test_date_param_one_day_behind_allowed(self, monkeypatch, client):
        import datetime as _dt
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        monkeypatch.setattr(dash, "_server_today", lambda: _dt.date(2026, 10, 2))
        req = make_request(
            headers={"Authorization": "Bearer mykey"},
            query_string="date=2026-10-01",
        )
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 200

    async def test_date_param_two_days_out_rejected(self, monkeypatch, client):
        import datetime as _dt
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        monkeypatch.setattr(dash, "_server_today", lambda: _dt.date(2026, 10, 2))
        req = make_request(
            headers={"Authorization": "Bearer mykey"},
            query_string="date=2026-09-30",
        )
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 400

    async def test_date_param_invalid_format_rejected(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(
            headers={"Authorization": "Bearer mykey"},
            query_string="date=October+2nd",
        )
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 400

    async def test_date_param_400_includes_cache_control(self, monkeypatch, client):
        import datetime as _dt
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        monkeypatch.setattr(dash, "_server_today", lambda: _dt.date(2026, 10, 2))
        req = make_request(
            headers={"Authorization": "Bearer mykey"},
            query_string="date=2020-01-01",
        )
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 400
        assert resp.headers.get("cache-control") == "no-store"

    async def test_garmin_called_with_requested_date(self, monkeypatch, client):
        import datetime as _dt
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        monkeypatch.setattr(dash, "_server_today", lambda: _dt.date(2026, 10, 2))
        req = make_request(
            headers={"Authorization": "Bearer mykey"},
            query_string="date=2026-10-01",
        )
        await dash._dashboard_handler(req)
        # The HRV call (one call, no loop) must use the requested date.
        client.get_hrv_data.assert_called_with("2026-10-01")
        # Sleep trend loops 7 days ending on the requested date; today (delta=0)
        # inside _fetch_sleep is 2026-10-01, so it must appear in calls.
        sleep_dates = [c[0][0] for c in client.get_sleep_data.call_args_list]
        assert "2026-10-01" in sleep_dates
        assert "2026-10-02" not in sleep_dates  # server today must not be used


# ---------------------------------------------------------------------------
# Cache-Control: no-store on every response
# ---------------------------------------------------------------------------

class TestCacheControlHeader:
    async def test_no_store_on_200(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        assert resp.headers.get("cache-control") == "no-store"

    async def test_no_store_on_401(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
        req = make_request(headers={"Authorization": "Bearer anything"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 401
        assert resp.headers.get("cache-control") == "no-store"

    async def test_no_store_on_preflight_204(self, monkeypatch):
        req = make_request(method="OPTIONS")
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 204
        assert resp.headers.get("cache-control") == "no-store"

    async def test_no_store_on_cached_200(self, monkeypatch, client):
        """Cache-Control: no-store must appear even when the response is served
        from the in-process cache (the header is browser/proxy-facing, not
        related to whether we cached internally)."""
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        await dash._dashboard_handler(req)  # populate cache
        resp = await dash._dashboard_handler(req)  # served from cache
        assert resp.headers.get("cache-control") == "no-store"


# ---------------------------------------------------------------------------
# Error payload not cached; clean payload cached
# ---------------------------------------------------------------------------

class TestCachingPolicy:
    async def test_error_payload_not_stored_in_cache(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        client.get_hrv_data.side_effect = RuntimeError("transient")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        assert resp.status_code == 200
        assert len(dash._cache) == 0

    async def test_clean_payload_stored_in_cache(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        await dash._dashboard_handler(req)
        assert len(dash._cache) == 1

    async def test_error_then_clean_refetches(self, monkeypatch, client):
        """After a partial failure (not cached), the next call hits Garmin again
        rather than serving the stale null."""
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        client.get_hrv_data.side_effect = RuntimeError("transient")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        await dash._dashboard_handler(req)  # error — not cached
        # Fix the error and call again
        client.get_hrv_data.side_effect = None
        await dash._dashboard_handler(req)
        # Second call must have re-fetched HRV (would be 1 if cache had served it)
        assert client.get_hrv_data.call_count == 2


# ---------------------------------------------------------------------------
# Shared get_stats call for body_battery + resting_hr
# ---------------------------------------------------------------------------

class TestSharedStats:
    async def test_get_stats_called_exactly_once(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        await dash._dashboard_handler(req)
        assert client.get_stats.call_count == 1

    async def test_both_sections_populated_from_one_call(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        import json
        body = json.loads(resp.body)
        assert body["body_battery"] is not None
        assert body["resting_hr"] is not None

    async def test_stats_failure_nulls_both_sections(self, monkeypatch, client):
        monkeypatch.setenv("DASHBOARD_API_KEY", "mykey")
        client.get_stats.side_effect = RuntimeError("stats down")
        req = make_request(headers={"Authorization": "Bearer mykey"})
        resp = await dash._dashboard_handler(req)
        import json
        body = json.loads(resp.body)
        assert body["body_battery"] is None
        assert body["resting_hr"] is None
        assert body["errors"]["body_battery"] == "RuntimeError"
        assert body["errors"]["resting_hr"] == "RuntimeError"
        # Other sections still fetched despite stats failure
        assert body["hrv"] is not None

    async def test_fetch_stats_returns_tuple(self, monkeypatch, client):
        import asyncio as _asyncio
        monkeypatch.setattr(
            dash, "_run_blocking", AsyncMock(side_effect=_sync_run_blocking)
        )
        bb, rhr = await dash._fetch_stats("2026-10-02")
        assert bb is not None
        assert rhr is not None
        assert "current" in bb
        assert "bpm" in rhr
