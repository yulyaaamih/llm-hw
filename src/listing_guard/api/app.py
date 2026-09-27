import asyncio
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import openai
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from listing_guard.common.media import stream_duration
from listing_guard.pipeline import OUT_DIR, ListingGuard
from listing_guard.text_moderation.model import DEFAULT_THRESHOLD, MODEL_ID, TextModerator

STATIC = Path(__file__).with_name("static")
MAX_PHOTOS = 10
MAX_REQUEST_MB = 200
MAX_MEDIA_S = 120
state: dict = {}


class ModerationRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0, le=1)


class ModerationResult(BaseModel):
    verdict: str
    reasons: list[str]
    scores: dict[str, float]


class ModerationResponse(BaseModel):
    model: str
    threshold: float
    results: list[ModerationResult]


class ListingResponse(BaseModel):
    id: str
    verdict: str
    reasons: list[str]
    review: list[str]
    transcripts: dict[str, str] = {}
    attributes: dict[str, str] | None = None
    attributes_error: str | None = None
    image_moderation: dict[str, dict] = {}
    text_moderation: dict[str, dict] = {}
    video: dict | None = None
    timings_s: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    guard = ListingGuard()
    for name in ("text", "image", "asr", "video"):
        await run_in_threadpool(getattr, guard, name)
    try:
        await run_in_threadpool(guard.vlm.predict, "warmup", Image.new("RGB", (448, 448), "white"))
    except openai.OpenAIError:
        pass
    state.update(guard=guard, lock=asyncio.Lock())
    yield
    state.clear()


app = FastAPI(title="Listing Guard", version="0.2.0", lifespan=lifespan)
OUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUT_DIR), name="files")


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > MAX_REQUEST_MB * 2**20:
        return JSONResponse({"detail": f"request larger than {MAX_REQUEST_MB} MB"}, status_code=413)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    guard: ListingGuard | None = state.get("guard")
    if guard is None:
        return {"status": "loading"}
    try:
        guard.vlm.client.with_options(timeout=2, max_retries=0).models.list()
        llm = "ok"
    except openai.OpenAIError:
        llm = "unavailable"
    return {"status": "ok", "device": guard.text.device, "llm": llm}


@app.post("/v1/moderate/text", response_model=ModerationResponse)
async def moderate(req: ModerationRequest) -> ModerationResponse:
    model: TextModerator = state["guard"].text
    async with state["lock"]:
        results = await run_in_threadpool(model.predict, req.texts, req.threshold)
    return ModerationResponse(model=MODEL_ID, threshold=req.threshold, results=results)


def save(upload: UploadFile | None, dst_dir: Path, stem: str) -> str | None:
    if upload is None or not upload.filename:
        return None
    dst = dst_dir / f"{stem}{Path(upload.filename).suffix.lower()}"
    with dst.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dst)


def check_image(path: str, name: str) -> None:
    try:
        Image.open(path).verify()
    except Exception as e:
        raise HTTPException(422, f"{name}: not an image") from e


def check_media(path: str, name: str, stream: str) -> None:
    duration = stream_duration(path, stream)
    if not duration:
        raise HTTPException(422, f"{name}: no readable {stream} stream")
    if duration > MAX_MEDIA_S:
        raise HTTPException(422, f"{name}: longer than {MAX_MEDIA_S} s")


def url(path: str | None) -> str | None:
    return f"/files/{Path(path).relative_to(OUT_DIR)}" if path else None


@app.post("/v1/listings/check", response_model=ListingResponse)
async def check_listing(
    title: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    photos: Annotated[list[UploadFile], File()] = (),
    audio: Annotated[UploadFile | None, File()] = None,
    video: Annotated[UploadFile | None, File()] = None,
) -> ListingResponse:
    photos = [p for p in photos if p.filename]
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(422, f"at most {MAX_PHOTOS} photos")
    listing_id = uuid.uuid4().hex[:8]
    text = "\n".join(t.strip() for t in (title, description) if t.strip())
    with tempfile.TemporaryDirectory() as tmp:
        upload_dir = Path(tmp)
        photo_paths = [save(p, upload_dir, f"photo{i}") for i, p in enumerate(photos, 1)]
        for i, p in enumerate(photo_paths, 1):
            check_image(p, f"photo {i}")
        audio_path, video_path = save(audio, upload_dir, "audio"), save(video, upload_dir, "video")
        if audio_path:
            check_media(audio_path, "audio", "audio")
        if video_path:
            check_media(video_path, "video", "video")
        if not (text or photo_paths or audio_path or video_path):
            raise HTTPException(422, "empty listing")
        async with state["lock"]:
            report = await run_in_threadpool(state["guard"].check, listing_id, text, photo_paths, audio_path,
                                             video_path, OUT_DIR / listing_id)
    if v := report.get("video"):
        v["blurred"], v["preview"] = url(v["blurred"]), url(v["preview"])
    return ListingResponse(**report)
