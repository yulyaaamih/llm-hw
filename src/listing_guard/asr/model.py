import numpy as np
from transformers import pipeline

from listing_guard.common.media import SAMPLE_RATE

MODEL_ID = "openai/whisper-small"


class SpeechRecognizer:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )

    def predict(self, wav: np.ndarray) -> dict:
        out = self.pipe(
            {"raw": wav, "sampling_rate": SAMPLE_RATE},
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )
        return {"text": out["text"].strip()}
