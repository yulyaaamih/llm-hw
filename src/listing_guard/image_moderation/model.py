from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
CLASSES = Path(__file__).with_name("classes.yaml")


def load_rules(path: Path) -> tuple[str, list[str], list[str], set[str]]:
    cfg = yaml.safe_load(path.read_text())
    prompts, prompt_category, forbidden = [], [], set()
    for group in ("forbidden", "allowed"):
        for category, category_prompts in cfg[group].items():
            if group == "forbidden":
                forbidden.add(category)
            prompts += category_prompts
            prompt_category += [category] * len(category_prompts)
    return cfg["prompt_template"], prompts, prompt_category, forbidden


class ImageModerator:
    def __init__(self, model_id: str = MODEL_ID, classes: Path = CLASSES, device: str = "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        template, self.prompts, self.prompt_category, forbidden = load_rules(classes)
        self.is_forbidden = torch.tensor([c in forbidden for c in self.prompt_category])
        texts = [template.format(p) for p in self.prompts]
        with torch.inference_mode():
            tokens = self.processor(text=texts, return_tensors="pt", padding=True).to(device)
            self.text_emb = normalize(self.model.get_text_features(**tokens).pooler_output)

    @torch.inference_mode()
    def predict(self, images: list[Image.Image]) -> list[dict]:
        pixels = self.processor(images=images, return_tensors="pt").to(self.device)
        emb = normalize(self.model.get_image_features(**pixels).pooler_output)
        probs = (self.model.logit_scale.exp() * emb @ self.text_emb.T).softmax(dim=-1).cpu()
        return [self._result(p) for p in probs]

    def _result(self, p: torch.Tensor) -> dict:
        i = int(p.argmax())
        return {
            "verdict": "not_ok" if self.is_forbidden[i] else "ok",
            "category": self.prompt_category[i],
            "prompt": self.prompts[i],
            "forbidden_score": round(float(p[self.is_forbidden].sum()), 4),
        }
