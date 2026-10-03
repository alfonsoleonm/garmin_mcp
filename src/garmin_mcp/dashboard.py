"""Whoop-style dashboard endpoint: GET /api/dashboard"""
import asyncio
import datetime
import hmac
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
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


async def _fetch_activities(date_str: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch activities for 7 days ending on date_str, grouped by date.

    Uses connectapi directly to avoid the library's auto-pagination loop,
    which can cause "result too large" errors on accounts with large histories.
    """
    end = datetime.date.fromisoformat(date_str)
    start_str = (end - datetime.timedelta(days=6)).isoformat()

    raw = await _run_blocking(
        lambda: _garmin_client.connectapi(
            _garmin_client.garmin_connect_activities,
            params={"startDate": start_str, "endDate": date_str, "start": "0", "limit": "100"},
        )
    )

    # Pre-fill all 7 days so missing days appear as empty lists.
    result: Dict[str, List[Dict[str, Any]]] = {
        (end - datetime.timedelta(days=i)).isoformat(): [] for i in range(7)
    }

    for a in raw or []:
        start_time = a.get("startTimeLocal", "")
        day = start_time[:10] if start_time else None
        if not day or day not in result:
            continue
        entry: Dict[str, Any] = {}
        duration = a.get("duration")
        if duration is not None:
            entry["duration_s"] = round(duration)
        avg_hr = a.get("averageHR")
        if avg_hr is not None:
            entry["average_hr"] = avg_hr
        max_hr = a.get("maxHR")
        if max_hr is not None:
            entry["max_hr"] = max_hr
        calories = a.get("calories")
        if calories is not None:
            entry["calories"] = round(calories)
        act_type = (a.get("activityType") or {}).get("typeKey")
        if act_type:
            entry["type"] = act_type
        aerobic_te = a.get("aerobicTrainingEffect")
        if aerobic_te is not None:
            entry["aerobic_te"] = aerobic_te
        anaerobic_te = a.get("anaerobicTrainingEffect")
        if anaerobic_te is not None:
            entry["anaerobic_te"] = anaerobic_te
        result[day].append(entry)

    return result


def _athlete_max_hr() -> int:
    """Return athlete max HR from DASHBOARD_MAX_HR env var (default 190, range 120–230).

    Falls back to 190 when the var is absent, non-integer, or out of range.
    """
    raw = os.environ.get("DASHBOARD_MAX_HR", "")
    if raw:
        try:
            val = int(raw)
            if 120 <= val <= 230:
                return val
        except ValueError:
            pass
    return 190


def _compute_strain_7d(
    activities_by_day: Dict[str, List[Dict[str, Any]]],
    resting_hr_bpm: Optional[int],
) -> Dict[str, Optional[float]]:
    # TRIMP (Training Impulse) per Banister (1991).
    # For each activity: trimp = duration_min × ratio × exp(1.92 × ratio)
    # where ratio = (avg_hr − resting_hr) / (HRMAX − resting_hr), clamped to [0, 1].
    # HRMAX is the athlete's max HR from DASHBOARD_MAX_HR env var (default 190).
    # Day total = Σ per-activity TRIMPs, rounded to 1 decimal.
    # Activities missing duration_s or average_hr are skipped.
    rhr = resting_hr_bpm if resting_hr_bpm and resting_hr_bpm > 0 else 60
    hrmax = _athlete_max_hr()
    if hrmax <= rhr:
        return {day: 0.0 for day in activities_by_day}
    result: Dict[str, Optional[float]] = {}
    for day, acts in activities_by_day.items():
        if not acts:
            result[day] = None
            continue
        total = 0.0
        for a in acts:
            duration_s = a.get("duration_s")
            avg_hr = a.get("average_hr")
            if duration_s is None or avg_hr is None:
                continue
            ratio = (avg_hr - rhr) / (hrmax - rhr)
            ratio = max(0.0, min(1.0, ratio))
            total += (duration_s / 60.0) * ratio * math.exp(1.92 * ratio)
        result[day] = round(total, 1) if total > 0 else None
    return result


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
        _safe("activities", _fetch_activities(date_str)),
    )

    # strain_7d is pure computation from activities + resting_hr — no IO.
    if sections.get("activities") is not None:
        try:
            rhr_section = sections.get("resting_hr")
            rhr_bpm = rhr_section.get("bpm") if isinstance(rhr_section, dict) else None
            sections["strain_7d"] = _compute_strain_7d(sections["activities"], rhr_bpm)
        except Exception as exc:
            sections["strain_7d"] = None
            errors["strain_7d"] = type(exc).__name__
    else:
        sections["strain_7d"] = None

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
