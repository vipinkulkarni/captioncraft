import os


def get_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_frame_count(duration_s: float) -> int:
    """Frame count from video duration, or a fixed FRAME_COUNT override."""
    fixed = os.environ.get("FRAME_COUNT", "").strip()
    if fixed:
        return max(int(fixed), 1)

    long_threshold = get_float_env("FRAME_LONG_DURATION_S", 60.0)
    if duration_s >= long_threshold:
        interval_s = max(get_float_env("FRAME_LONG_INTERVAL_S", 3.0), 0.5)
        min_frames = max(get_int_env("FRAME_LONG_COUNT_MIN", 12), 1)
        max_frames = max(get_int_env("FRAME_LONG_COUNT_MAX", 28), min_frames)
    else:
        interval_s = max(get_float_env("FRAME_INTERVAL_S", 4.0), 0.5)
        min_frames = max(get_int_env("FRAME_COUNT_MIN", 8), 1)
        max_frames = max(get_int_env("FRAME_COUNT_MAX", 24), min_frames)

    if duration_s <= 0:
        return min_frames

    computed = round(duration_s / interval_s)
    return max(min_frames, min(max_frames, computed))


def resolve_frame_width(duration_s: float = 0.0) -> int:
    """Wider frames on short clips; keep narrow width on long clips for speed."""
    base = max(get_int_env("FRAME_WIDTH", 384), 64)
    short = max(get_int_env("FRAME_SHORT_WIDTH", base), 64)
    long_threshold = get_float_env("FRAME_LONG_DURATION_S", 60.0)
    if duration_s > 0 and duration_s < long_threshold:
        return short
    return base


def get_frame_config(duration_s: float = 0.0) -> tuple[int, int]:
    return resolve_frame_count(duration_s), resolve_frame_width(duration_s)
