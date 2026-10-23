"""
Comic-Lab Version and Auto-Update Engine
"""
import re
from typing import Tuple

APP_VERSION = "1.0.1"
APP_NAME = "Comic-Lab Studio"
DEFAULT_UPDATE_URL = "https://comic-lab-backend.onrender.com/api/version"

def parse_version(v_str: str) -> Tuple[int, ...]:
    """Parses a semantic version string (e.g. '1.0.1', 'v1.2.0') into an integer tuple."""
    if not v_str:
        return (0, 0, 0)
    nums = re.findall(r"\d+", str(v_str))
    return tuple(map(int, nums)) if nums else (0, 0, 0)

def is_newer_version(remote_v: str, local_v: str = APP_VERSION) -> bool:
    """Returns True if remote_v is strictly greater than local_v."""
    r_tuple = parse_version(remote_v)
    l_tuple = parse_version(local_v)
    return r_tuple > l_tuple
