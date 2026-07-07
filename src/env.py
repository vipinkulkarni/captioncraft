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


def get_frame_config() -> tuple[int, int]:
    frame_count = max(get_int_env("FRAME_COUNT", 8), 1)
    frame_width = max(get_int_env("FRAME_WIDTH", 512), 64)
    return frame_count, frame_width
