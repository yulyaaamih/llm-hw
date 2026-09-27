import io
import subprocess
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image
from tqdm import tqdm

from listing_guard.common import DATA_DIR

FASHION_HIRES = "datasets/benitomartin/fashion-product-images-small-900x1200/data"
FASHION_IDS = [41602, 36426, 5884]
OPEN_IMAGES_URL = "https://s3.amazonaws.com/open-images-dataset/validation/{}.jpg"
FORBIDDEN_PHOTO = "0c87545ba250f18c"
PEXELS_URL = "https://www.pexels.com/download/video/{}/"
PEXELS_ID = 4057411
VIDEO_SECONDS = 15


def fetch(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dst)
    return dst


def fashion_photos(ids: list[int], out: Path) -> None:
    want = {i for i in ids if not (out / f"{i}.jpg").exists()}
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for path in sorted(fs.ls(FASHION_HIRES, detail=False)):
        if not want:
            break
        pf = pq.ParquetFile(fs.open(path))
        for rg in range(pf.num_row_groups):
            if not want & set(pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()):
                continue
            for row in pf.read_row_group(rg, columns=["id", "image"]).to_pylist():
                if row["id"] in want:
                    Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(out / f"{row['id']}.jpg")
                    want.discard(row["id"])


def pexels_clip(video_id: int, dst: Path) -> None:
    if dst.exists():
        return
    raw = fetch(PEXELS_URL.format(video_id), dst.with_name(f"{video_id}_raw.mp4"))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-t", str(VIDEO_SECONDS),
         "-vf", "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)'", "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(dst)],
        check=True,
    )
    raw.unlink()


def main() -> None:
    fashion_photos(FASHION_IDS, DATA_DIR / "demo")
    fetch(OPEN_IMAGES_URL.format(FORBIDDEN_PHOTO), DATA_DIR / "open_images/forbidden/firearms" / f"{FORBIDDEN_PHOTO}.jpg")
    pexels_clip(PEXELS_ID, DATA_DIR / "videos" / f"pexels_{PEXELS_ID}.mp4")
    print(f"demo data in {DATA_DIR}")


if __name__ == "__main__":
    main()
