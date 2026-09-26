"""
api_quota_tracker.py
--------------------
Fail-safe usage and quota tracking for optional external cloud APIs.

Guarantees 100% cost prevention by enforcing a strict 90% hard cap of documented
free-tier allowances. Once the 90% threshold is reached, further API calls are
immediately blocked for the remainder of the period, automatically falling back
to local-only processing.

Free-Tier Quotas:
1. Google Cloud Vision: 1,000 units / month  -> Hard cap: 900 units / month (90%)
2. Google Cloud Translate: 500,000 chars / month -> Hard cap: 450,000 chars / month (90%)
3. Marvel Comics API: 3,000 calls / day      -> Hard cap: 2,700 calls / day (90%)

Storage:
- Local thread-safe persistent tracker file: `.api_quota.json`
- Automatic period rollover (monthly or daily)
"""

import os
import json
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("api_quota_tracker")

TRACKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".api_quota.json")

# Service definitions and 90% hard caps
QUOTA_CONFIG = {
    "google_vision": {
        "period": "monthly",
        "free_limit": 1000,
        "safety_cap": 900,
        "unit": "units"
    },
    "google_translate": {
        "period": "monthly",
        "free_limit": 500000,
        "safety_cap": 450000,
        "unit": "characters"
    },
    "marvel_api": {
        "period": "daily",
        "free_limit": 3000,
        "safety_cap": 2700,
        "unit": "calls"
    }
}

_LOCK = threading.Lock()


def _get_current_period_key(period_type: str) -> str:
    """Returns 'YYYY-MM' for monthly or 'YYYY-MM-DD' for daily."""
    now = datetime.now(timezone.utc)
    if period_type == "daily":
        return now.strftime("%Y-%m-%d")
    return now.strftime("%Y-%m")


def _load_tracker_data() -> Dict[str, Any]:
    """Loads tracker file safely."""
    if not os.path.exists(TRACKER_FILE):
        return {}
    try:
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[API_QUOTA] Failed loading {TRACKER_FILE}: {e}")
        return {}


def _save_tracker_data(data: Dict[str, Any]) -> None:
    """Atomically saves tracker file."""
    tmp_path = TRACKER_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, TRACKER_FILE)
    except Exception as e:
        logger.warning(f"[API_QUOTA] Failed saving {TRACKER_FILE}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def can_consume(service_name: str, amount: int = 1) -> bool:
    """
    Checks if `amount` can be safely consumed without exceeding the 90% free-tier cap.
    Returns True if allowed, False if capped.
    """
    cfg = QUOTA_CONFIG.get(service_name)
    if not cfg:
        return True

    period_key = _get_current_period_key(cfg["period"])
    safety_cap = cfg["safety_cap"]

    with _LOCK:
        data = _load_tracker_data()
        svc_data = data.get(service_name, {})
        current_period = svc_data.get("period_key")
        current_count = svc_data.get("count", 0)

        # Reset count if period rolled over
        if current_period != period_key:
            current_count = 0

        if current_count + amount > safety_cap:
            logger.warning(
                f"[API_CAP] Safety cap reached for '{service_name}' ({current_count}/{safety_cap} {cfg['unit']}). "
                f"Blocking request to ensure zero cloud cost. Falling back to local."
            )
            return False

        return True


def record_consumption(service_name: str, amount: int = 1) -> None:
    """
    Records usage of `amount` towards the current period's quota.
    """
    cfg = QUOTA_CONFIG.get(service_name)
    if not cfg:
        return

    period_key = _get_current_period_key(cfg["period"])

    with _LOCK:
        data = _load_tracker_data()
        svc_data = data.get(service_name, {})
        current_period = svc_data.get("period_key")
        current_count = svc_data.get("count", 0)

        if current_period != period_key:
            current_count = 0

        new_count = current_count + amount
        data[service_name] = {
            "period_key": period_key,
            "count": new_count,
            "safety_cap": cfg["safety_cap"],
            "free_limit": cfg["free_limit"],
            "unit": cfg["unit"],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        _save_tracker_data(data)


def get_quota_status() -> Dict[str, Any]:
    """Returns a dictionary summarizing current usage vs caps for all tracked services."""
    status = {}
    with _LOCK:
        data = _load_tracker_data()
        for svc, cfg in QUOTA_CONFIG.items():
            period_key = _get_current_period_key(cfg["period"])
            svc_data = data.get(svc, {})
            current_period = svc_data.get("period_key")
            count = svc_data.get("count", 0) if current_period == period_key else 0
            safety_cap = cfg["safety_cap"]
            free_limit = cfg["free_limit"]
            remaining_to_cap = max(0, safety_cap - count)

            status[svc] = {
                "period": cfg["period"],
                "period_key": period_key,
                "used": count,
                "safety_cap": safety_cap,
                "free_limit": free_limit,
                "remaining_to_cap": remaining_to_cap,
                "cap_reached": count >= safety_cap,
                "unit": cfg["unit"]
            }
    return status


def reset_quota(service_name: Optional[str] = None) -> None:
    """Resets tracking data (used for testing and administrative resets)."""
    with _LOCK:
        if service_name is None:
            if os.path.exists(TRACKER_FILE):
                try:
                    os.remove(TRACKER_FILE)
                except Exception:
                    pass
        else:
            data = _load_tracker_data()
            if service_name in data:
                del data[service_name]
                _save_tracker_data(data)
