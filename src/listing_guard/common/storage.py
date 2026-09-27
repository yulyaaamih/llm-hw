import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse


def _read_file(uri: str) -> bytes:
    return Path(uri.removeprefix("file://")).read_bytes()


READERS: dict[str, Callable[[str], bytes]] = {"file": _read_file}


def scheme(uri: str) -> str:
    return urlparse(uri).scheme or "file"


def read_bytes(uri: str) -> bytes:
    s = scheme(uri)
    if s not in READERS:
        raise ValueError(f"unsupported storage scheme {s!r}, available: {sorted(READERS)}")
    return READERS[s](uri)


@contextmanager
def _temp_file(data: bytes, suffix: str = "") -> Iterator[Path]:
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        yield Path(f.name)


@contextmanager
def local_file(uri: str) -> Iterator[Path]:
    if scheme(uri) == "file":
        yield Path(uri.removeprefix("file://"))
        return
    with _temp_file(read_bytes(uri), Path(urlparse(uri).path).suffix) as path:
        yield path
