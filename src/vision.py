
"""Core computer-vision logic: object detection, OCR, distance estimation,
direction detection, and image-quality checks."""
import os
import cv2
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "yolov8s.onnx"
)
# The 80 COCO class names used by YOLOv8.
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Approximate real-world heights in metres.
REAL_HEIGHT_M = {
    "person": 1.7,
    "chair": 0.9,
    "car": 1.5,
    "bus": 3.0,
    "truck": 3.0,
    "bicycle": 1.0,
    "motorcycle": 1.1,
    "dining table": 0.75,
    "couch": 0.9,
    "bed": 0.6,
    "tv": 0.5,
    "bottle": 0.25,
    "cup": 0.1,
    "laptop": 0.25,
    "refrigerator": 1.7,
    "door": 2.0,
    "bench": 0.9,
    "dog": 0.5,
    "cat": 0.3,
    "stop sign": 0.75,
    "traffic light": 0.9,
    "book": 0.25, "cell phone": 0.15, "backpack": 0.45, "handbag": 0.3,
    "umbrella": 0.9, "potted plant": 0.5, "clock": 0.3, "keyboard": 0.04,
    "mouse": 0.04, "remote": 0.18, "bowl": 0.08, "vase": 0.3,
    "microwave": 0.3, "oven": 0.6, "sink": 0.2, "toilet": 0.4,
}

# Approximate focal length in pixels; calibrate for your camera.
FOCAL_PX = 600.0


class Detector:
    def __init__(self, model_path=MODEL_PATH, conf=0.25, nms=0.45, size=960):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf = conf
        self.nms = nms
        self.size = size

    def detect(self, frame):
        h, w = frame.shape[:2]

        # Letterbox: keep aspect ratio, pad the rest with grey.
        scale = min(self.size / w, self.size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()[0]
        if out.shape[0] == 84:
            out = out.T

        cls_scores = out[:, 4:]
        ids_raw = np.argmax(cls_scores, axis=1)
        scores_raw = cls_scores[np.arange(len(cls_scores)), ids_raw]
        mask = scores_raw >= self.conf
        if not np.any(mask):
            return []

        b = out[:, :4][mask]
        scores = scores_raw[mask].astype(float).tolist()
        ids = ids_raw[mask].astype(int).tolist()

        # Undo the letterbox to get coordinates on the original frame.
        x = np.clip((b[:, 0] - b[:, 2] / 2 - px) / scale, 0, w - 1)
        y = np.clip((b[:, 1] - b[:, 3] / 2 - py) / scale, 0, h - 1)
        bw = b[:, 2] / scale
        bh = b[:, 3] / scale
        boxes = np.stack([x, y, bw, bh], axis=1).astype(int).tolist()

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms)
        results = []
        for i in np.array(keep).flatten() if len(keep) else []:
            if ids[i] < len(CLASSES):
                results.append({
                    "label": CLASSES[ids[i]],
                    "confidence": round(scores[i], 2),
                    "box": boxes[i],
                })
        return results


def estimate_distance(label, box, focal_px=FOCAL_PX):
    """Estimate distance using the pinhole camera model."""

    real_h = REAL_HEIGHT_M.get(label)
    pixel_h = box[3] if len(box) == 4 else 0

    if not real_h or pixel_h <= 0:
        return None

    return round(real_h * focal_px / pixel_h, 1)


def direction(box, frame_width):
    """Determine whether an object is left, ahead, or right."""

    x, y, box_w, box_h = box
    cx = x + box_w / 2

    if cx < frame_width * 0.33:
        return "left"

    if cx > frame_width * 0.66:
        return "right"

    return "ahead"


def image_quality(frame):
    """Detect dark or blurry frames."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []

    if brightness < 50:
        warnings.append("low_light")

    if blur < 60:
        warnings.append("blurry")

    return {
        "brightness": round(brightness, 1),
        "sharpness": round(blur, 1),
        "warnings": warnings,
    }


def read_text(frame):
    """Extract text using Tesseract OCR."""

    try:
        import pytesseract

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        text = pytesseract.image_to_string(binary)

        return " ".join(text.split())

    except Exception:
        return ""


def analyze(detector, frame, do_ocr=True):
    """Run object detection, distance estimation, OCR, and quality checks."""

    _, frame_width = frame.shape[:2]

    quality = image_quality(frame)
    objects = []

    for detection in detector.detect(frame):
        box = detection["box"]

        objects.append({
            "label": detection["label"],
            "confidence": detection["confidence"],
            "distance_m": estimate_distance(
                detection["label"], box
            ),
            "direction": direction(box, frame_width),
            "box": box,
        })

    # Sort objects by estimated distance when available.
    objects.sort(
        key=lambda obj: (
            obj["distance_m"] is None,
            obj["distance_m"] or 0,
        )
    )

    text = read_text(frame) if do_ocr else ""

    return {
        "objects": objects,
        "text": text,
        "quality": quality,
    }


def draw(frame, objects):
    """Draw object bounding boxes and distance labels on a frame."""

    for obj in objects:
        x, y, w, h = obj["box"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        distance = (
            f" {obj['distance_m']}m"
            if obj["distance_m"] is not None
            else ""
        )

        label = f"{obj['label']}{distance}"

        cv2.putText(
            frame,
            label,
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    return frame
