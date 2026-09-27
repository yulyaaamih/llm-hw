import base64
import io
import json
import os
from pathlib import Path

import yaml
from openai import OpenAI
from PIL import Image

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b")
ATTRIBUTES = yaml.safe_load(Path(__file__).with_name("attributes.yaml").read_text())
SYSTEM = (
    "You are a catalog assistant of an online marketplace. Fill in the product card attributes "
    "from the seller's listing. Use only the allowed values. Answer with a single JSON object."
)


SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string", "enum": v} for k, v in ATTRIBUTES.items()},
    "required": list(ATTRIBUTES),
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "attributes", "schema": SCHEMA, "strict": True}}
INSTRUCTIONS = (
    "Allowed values:\n" + "\n".join(f"- {k}: {', '.join(v)}" for k, v in ATTRIBUTES.items())
    + f"\n\nReturn JSON with keys: {', '.join(ATTRIBUTES)}."
)


def image_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def messages(text: str | None, image: Image.Image | None) -> list[dict]:
    parts = []
    if image is not None:
        parts.append({"type": "image_url", "image_url": {"url": image_url(image)}})
    listing = f'Listing text: "{text}"' if text else "The listing has only a photo."
    parts.append({"type": "text", "text": f"{listing}\n\n{INSTRUCTIONS}"})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}]


class AttributeExtractor:
    def __init__(self, base_url: str = BASE_URL, model: str = MODEL):
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "EMPTY"))
        self.model = model

    def predict(self, text: str | None = None, image: Image.Image | None = None) -> dict[str, str] | None:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages(text, image), temperature=0, max_tokens=200,
            response_format=RESPONSE_FORMAT,
        )
        try:
            attrs = json.loads(r.choices[0].message.content or "")
        except json.JSONDecodeError:
            return None
        valid = isinstance(attrs, dict) and all(attrs.get(k) in v for k, v in ATTRIBUTES.items())
        return attrs if valid else None
