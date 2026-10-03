"""Whoop-style dashboard endpoint: GET /api/dashboard"""
import asyncio
import datetime
import hmac
import os
import time
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_garmin_client = None

_ALLOWED_ORIGINS: frozenset = frozenset(
    {"https://alfonsoleonm.github.io", "http://localhost:5173"}
)
_CACHE_TTL: float = 90.0
# keyed by date string (YYYY-MM-DD) → (payload_dict, monotonic_timestamp)
_cache: Dict[str, tuple] = {}


def configure(client) -> None:
    global _garmin_client
    _garmin_client = client


def _server_today() -> datetime.date:
    """Return the current date in the configured timezone (DASHBOARD_TZ env var).

    Falls back to America/Mexico_City when the env var is absent or names an
    unknown timezone, rather than crashing the whole request.
    """
    tz_name = os.environ.get("DASHBOARD_TZ", "America/Mexico_City")
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, KeyError):
        tz = ZoneInfo("America/Mexico_City")
    return datetime.datetime.now(tz).date()


def _cors_headers(origin: str) -> Dict[str, str]:
    if origin in _ALLOWED_ORIGINS:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }
    return {}


def _response_headers(origin: str) -> Dict[str, str]:
    """Base headers for every response from this route.

    Cache-Control: no-store is unconditional — health data must never be
    stored in browser caches, proxies, or CDNs regardless of origin.
    """
    headers: Dict[str, str] = {"Cache-Control": "no-store"}
    headers.update(_cors_headers(origin))
    return headers


def _authorized(request: Request) -> bool:
    """Return True iff the request carries a valid Bearer token.

    Fails closed when DASHBOARD_API_KEY is unset so a misconfigured
    deployment doesn't accidentally expose data.
    """
    key = os.environ.get("DASHBOARD_API_KEY", "")
    if not key:
        return False
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    candidate = auth[7:]
    return hmac.compare_digest(candidate.encode(), key.encode())


async def _run_blocking(fn, *args: Any) -> Any:
    return await asyncio.to_thread(fn, *args)


# ---------------------------------------------------------------------------
# Section fetchers — each raises on failure; caller catches per-section
# ---------------------------------------------------------------------------

async def _fetch_sleep(date_str: str) -> Dict[str, Any]:
    from garmin_mcp.health_wellness import _extract_sleep_summary

    end = datetime.date.fromisoformat(date_str)
    trend = []
    today_summary = None

    for delta in range(6, -1, -1):  # oldest first
        d = (end - datetime.timedelta(days=delta)).isoformat()
        try:
            raw = await _run_blocking(_garmin_client.get_sleep_data, d)
            if raw:
                entry_data = _extract_sleep_summary(raw)
                trend.append({"date": d, **entry_data})
                if delta == 0:
                    today_summary = entry_data
        except Exception:
            pass  # individual nights are best-effort within the trend

    return {"today": today_summary, "trend_7d": trend}


async def _fetch_hrv(date_str: str) -> Optional[Dict[str, Any]]:
    raw = await _run_blocking(_garmin_client.get_hrv_data, date_str)
    if not raw:
        return None
    summary = raw.get("hrvSummary") or {}
    baseline = summary.get("baseline") or {}
    return {
        k: v
        for k, v in {
            "last_night_avg_ms": summary.get("lastNightAvg"),
            "last_night_5min_high_ms": summary.get("lastNight5MinHigh"),
            "weekly_avg_ms": summary.get("weeklyAvg"),
            "status": summary.get("status"),
            "feedback": summary.get("feedbackPhrase"),
            "baseline_balanced_low": baseline.get("balancedLow"),
            "baseline_balanced_upper": baseline.get("balancedUpper"),
        }.items()
        if v is not None
    }


async def _fetch_stats(
    date_str: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Call get_stats once; return (body_battery, resting_hr).

    Keeping both sections together means one Garmin round-trip per request
    instead of two, and they fail together if the endpoint is unavailable.
    """
    stats = await _run_blocking(_garmin_client.get_stats, date_str)
    if not stats:
        return None, None

    bb: Optional[Dict[str, Any]] = {
        k: v
        for k, v in {
            "current": stats.get("bodyBatteryMostRecentValue"),
            "highest": stats.get("bodyBatteryHighestValue"),
            "lowest": stats.get("bodyBatteryLowestValue"),
            "charged": stats.get("bodyBatteryChargedValue"),
            "drained": stats.get("bodyBatteryDrainedValue"),
        }.items()
        if v is not None
    } or None

    rhr: Optional[Dict[str, Any]] = {
        k: v
        for k, v in {
            "bpm": stats.get("restingHeartRate"),
            "last_7d_avg_bpm": stats.get("lastSevenDaysAvgRestingHeartRate"),
        }.items()
        if v is not None
    } or None

    return bb, rhr


async def _fetch_training_readiness(date_str: str) -> Optional[Dict[str, Any]]:
    raw = await _run_blocking(_garmin_client.get_training_readiness, date_str)
    if not raw:
        return None
    items = raw if isinstance(raw, list) else [raw]
    if not items:
        return None
    latest = items[-1] if isinstance(items[-1], dict) else {}
    result = {
        k: v
        for k, v in {
            "score": latest.get("score"),
            "level": latest.get("level"),
            "feedback": latest.get("feedbackShort"),
            "sleep_score": latest.get("sleepScore"),
            "hrv_weekly_avg": latest.get("hrvWeeklyAverage"),
            "acute_load": latest.get("acuteLoad"),
        }.items()
        if v is not None
    }
    return result or None


async def _fetch_training_load(date_str: str) -> Optional[Dict[str, Any]]:
    raw = await _run_blocking(_garmin_client.get_training_status, date_str)
    if not raw or not isinstance(raw, dict):
        return None
    recent = raw.get("mostRecentTrainingStatus") or {}
    latest_data = (
        recent.get("latestTrainingStatusData") or {}
        if isinstance(recent, dict)
        else {}
    )
    device_data: Dict[str, Any] = {}
    if isinstance(latest_data, dict):
        for data in latest_data.values():
            if isinstance(data, dict) and data:
                device_data = data
                break
    acwr = device_data.get("acuteTrainingLoadDTO") or {}
    if not isinstance(acwr, dict):
        acwr = {}
    result = {
        k: v
        for k, v in {
            "training_status": device_data.get("trainingStatus"),
            "feedback": device_data.get("trainingStatusFeedbackPhrase"),
            "acute_load": acwr.get("dailyTrainingLoadAcute"),
            "chronic_load": acwr.get("dailyTrainingLoadChronic"),
            "load_ratio": acwr.get("dailyAcuteChronicWorkloadRatio"),
            "acwr_status": acwr.get("acwrStatus"),
        }.items()
        if v is not None
    }
    return result or None


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

async def _dashboard_handler(request: Request) -> Response:
    origin = request.headers.get("origin", "")
    hdrs = _response_headers(origin)

    # Handle CORS preflight before auth so browsers can discover allowed headers.
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=hdrs)

    if not _authorized(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401, headers=hdrs)

    # Resolve the target date: server today in the configured timezone, or an
    # explicit ?date= override limited to ±1 day so it can't be used to walk
    # through historical data without a time limit.
    server_today = _server_today()
    raw_date = request.query_params.get("date")
    if raw_date is not None:
        try:
            requested = datetime.date.fromisoformat(raw_date)
        except ValueError:
            return JSONResponse(
                {"error": "Invalid date; use YYYY-MM-DD"},
                status_code=400,
                headers=hdrs,
            )
        if abs((requested - server_today).days) > 1:
            return JSONResponse(
                {"error": "date must be within 1 day of server today"},
                status_code=400,
                headers=hdrs,
            )
        date_str = requested.isoformat()
    else:
        date_str = server_today.isoformat()

    cached = _cache.get(date_str)
    if cached is not None:
        payload, ts = cached
        if time.monotonic() - ts < _CACHE_TTL:
            return JSONResponse(payload, headers=hdrs)

    errors: Dict[str, str] = {}
    sections: Dict[str, Any] = {}

    async def _safe(name: str, coro) -> None:
        try:
            sections[name] = await coro
        except Exception as exc:
            sections[name] = None
            # Only the exception class name — no message, no stack trace.
            errors[name] = type(exc).__name__

    async def _do_stats() -> None:
        # body_battery and resting_hr share a single get_stats call so Garmin
        # only sees one request for this pair of sections.
        try:
            bb, rhr = await _fetch_stats(date_str)
            sections["body_battery"] = bb
            sections["resting_hr"] = rhr
        except Exception as exc:
            err = type(exc).__name__
            sections["body_battery"] = None
            sections["resting_hr"] = None
            errors["body_battery"] = err
            errors["resting_hr"] = err

    await asyncio.gather(
        _safe("sleep", _fetch_sleep(date_str)),
        _safe("hrv", _fetch_hrv(date_str)),
        _do_stats(),
        _safe("training_readiness", _fetch_training_readiness(date_str)),
        _safe("training_load", _fetch_training_load(date_str)),
    )

    payload: Dict[str, Any] = {"date": date_str, **sections}
    if errors:
        payload["errors"] = errors

    # Only cache clean payloads: a partial response has stale nulls that
    # could be served to the next caller even after the Garmin issue resolves.
    if "errors" not in payload:
        _cache[date_str] = (payload, time.monotonic())

    return JSONResponse(payload, headers=hdrs)


def register_route(fastmcp) -> None:
    @fastmcp.custom_route("/api/dashboard", methods=["GET", "OPTIONS"])
    async def dashboard(request: Request) -> Response:
        return await _dashboard_handler(request)
