from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from listing_guard.common import DATA_DIR

WEIGHTS = "yolo11s.pt"
CONF = 0.1
PERSON = 0


def blur_boxes(frame: np.ndarray, boxes: np.ndarray, pad: float = 0.05) -> np.ndarray:
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = int(max(0, x1 - dx)), int(max(0, y1 - dy))
        x2, y2 = int(min(w, x2 + dx)), int(min(h, y2 + dy))
        if x2 <= x1 or y2 <= y1:
            continue
        k = max(31, (min(x2 - x1, y2 - y1) // 4) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)
    return frame


class VideoAnonymizer:
    def __init__(self, weights: str = WEIGHTS, device: str = "cpu"):
        path = DATA_DIR / "weights" / weights
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(path))
        self.device = device

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.predict(frame, conf=CONF, device=self.device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)

    def predict(self, video: Path, out_path: Path | None = None, preview: Path | None = None) -> dict:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size) if out_path else None
        frames, persons_frames, max_persons, objects = 0, 0, 0, Counter()
        best_preview = (0, None, None)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, classes = self.detect(frame)
            persons = boxes[classes == PERSON]
            frames += 1
            persons_frames += int(len(persons) > 0)
            max_persons = max(max_persons, len(persons))
            objects.update({self.model.names[c] for c in classes})
            if preview is not None and len(persons) > best_preview[0]:
                best_preview = (len(persons), frame.copy(), persons)
            frame = blur_boxes(frame, persons)
            if writer:
                writer.write(frame)
        cap.release()
        if writer:
            writer.release()
        if preview is not None and best_preview[1] is not None:
            _, raw, persons = best_preview
            cv2.imwrite(str(preview), np.hstack([raw, blur_boxes(raw.copy(), persons)]))
        return {
            "frames": frames,
            "fps": round(fps, 2),
            "persons_frames": persons_frames,
            "max_persons_in_frame": max_persons,
            "objects": dict(objects.most_common()),
        }
