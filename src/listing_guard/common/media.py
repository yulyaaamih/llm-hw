import io
import json
import subprocess

import numpy as np
from PIL import Image

from listing_guard.common.storage import local_file, read_bytes

SAMPLE_RATE = 16000


def load_audio(uri: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    with local_file(uri) as path:
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
               "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def load_image(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(read_bytes(uri))).convert("RGB")


def stream_duration(uri: str, stream: str) -> float | None:
    with local_file(uri) as path:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration:format=duration",
               "-of", "json", str(path)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode:
        return None
    info = json.loads(out.stdout)
    streams = [st for st in info.get("streams", []) if st.get("codec_type") == stream]
    if not streams:
        return None
    return float(streams[0].get("duration") or info.get("format", {}).get("duration") or 0)
