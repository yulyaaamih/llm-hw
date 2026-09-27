import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import openai
from PIL import Image

from listing_guard.asr.model import SpeechRecognizer
from listing_guard.common import RESULTS_DIR
from listing_guard.common.media import load_audio, load_image
from listing_guard.common.storage import local_file
from listing_guard.image_moderation.model import ImageModerator
from listing_guard.text_moderation.model import TextModerator
from listing_guard.video_detection.model import VideoAnonymizer
from listing_guard.vlm_attributes.model import AttributeExtractor

REJECT_THRESHOLD = 0.96
REVIEW_THRESHOLD = 0.73
OUT_DIR = RESULTS_DIR / "demo"


def to_h264(src: Path, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    src.unlink()
    return dst


@contextmanager
def timer(timings: dict[str, float], step: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    timings[step] = round(time.perf_counter() - start, 3)


class ListingGuard:
    @cached_property
    def asr(self) -> SpeechRecognizer:
        return SpeechRecognizer()

    @cached_property
    def text(self) -> TextModerator:
        return TextModerator()

    @cached_property
    def image(self) -> ImageModerator:
        return ImageModerator()

    @cached_property
    def video(self) -> VideoAnonymizer:
        return VideoAnonymizer()

    @cached_property
    def vlm(self) -> AttributeExtractor:
        return AttributeExtractor()

    def check(self, listing_id: str, text: str = "", photos: Sequence[str] = (), audio: str | None = None,
              video: str | None = None, out_dir: Path = OUT_DIR) -> dict:
        report = {"id": listing_id, "reasons": [], "review": [], "timings_s": {}}
        speech = self._speech(report, {"audio": audio, "video": video})
        texts = {"text": text} if text else {}
        texts |= {f"{src}_transcript": t for src, t in speech.items()}
        self._moderate_text(report, texts)
        images = self._moderate_images(report, photos)
        if video:
            self._blur_video(report, listing_id, video, out_dir)
        report["verdict"] = "not_ok" if report["reasons"] else "review" if report["review"] else "ok"
        if report["verdict"] != "not_ok" and (text or images):
            self._attributes(report, text, images)
        return report

    def transcribe(self, uri: str) -> str | None:
        try:
            wav = load_audio(uri)
        except subprocess.CalledProcessError:
            return None
        return self.asr.predict(wav)["text"] or None

    def _speech(self, report: dict, media: dict[str, str | None]) -> dict[str, str]:
        media = {src: uri for src, uri in media.items() if uri}
        if not media:
            return {}
        with timer(report["timings_s"], "asr"):
            speech = {src: t for src, uri in media.items() if (t := self.transcribe(uri))}
        if speech:
            report["transcripts"] = speech
        return speech

    def _moderate_text(self, report: dict, texts: dict[str, str]) -> None:
        if not texts:
            return
        with timer(report["timings_s"], "text_moderation"):
            results = self.text.predict(list(texts.values()), REVIEW_THRESHOLD)
        report["text_moderation"] = dict(zip(texts, results))
        for src, r in report["text_moderation"].items():
            for label in r["reasons"]:
                score = r["scores"][label]
                target = report["reasons"] if score >= REJECT_THRESHOLD else report["review"]
                target.append(f"{src}: {label} ({score:.2f})")

    def _moderate_images(self, report: dict, photos: Sequence[str]) -> list[Image.Image]:
        if not photos:
            return []
        with timer(report["timings_s"], "image_moderation"):
            images = [load_image(p) for p in photos]
            results = self.image.predict(images)
        report["image_moderation"] = {f"photo {i}": r for i, r in enumerate(results, 1)}
        report["reasons"] += [f"photo {i}: forbidden item {r['category']} ({r['forbidden_score']:.2f})"
                              for i, r in enumerate(results, 1) if r["verdict"] == "not_ok"]
        return images

    def _blur_video(self, report: dict, listing_id: str, video: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw, preview = out_dir / f"{listing_id}_raw.mp4", out_dir / f"{listing_id}_preview.jpg"
        with timer(report["timings_s"], "video"), local_file(video) as path:
            info = self.video.predict(path, raw, preview)
        blurred = to_h264(raw, out_dir / f"{listing_id}_blurred.mp4")
        report["video"] = {**info, "blurred": str(blurred), "preview": str(preview) if preview.exists() else None}

    def _attributes(self, report: dict, text: str, images: list[Image.Image]) -> None:
        try:
            with timer(report["timings_s"], "attributes"):
                report["attributes"] = self.vlm.predict(text or None, images[0] if images else None)
        except openai.OpenAIError as e:
            report["attributes"] = None
            report["attributes_error"] = f"LLM request failed: {type(e).__name__}"
