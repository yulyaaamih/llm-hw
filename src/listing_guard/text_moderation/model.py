import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "unitary/toxic-bert"
DEFAULT_THRESHOLD = 0.5


class TextModerator:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.inference_mode()
    def scores(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt").to(self.device)
        return torch.sigmoid(self.model(**batch).logits).float().cpu()

    def predict(self, texts: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
        out = []
        for row in self.scores(texts).tolist():
            scores = {label: round(p, 4) for label, p in zip(self.labels, row)}
            flagged = [label for label, p in scores.items() if p >= threshold]
            out.append({"verdict": "not_ok" if flagged else "ok", "reasons": flagged, "scores": scores})
        return out
