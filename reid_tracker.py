import os
import json
import threading
from datetime import datetime, date

import cv2
import numpy as np

# Prevent cv2's internal thread pool from fighting with PyTorch's for CPU threads. 
cv2.setNumThreads(1)

REID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
REID_DB_PATH = os.path.join(REID_DIR, "people_db.json")
EMBEDDINGS_PATH = os.path.join(REID_DIR, "embeddings.json")

# Pretrained person re-identification model (Tencent Youtu, via OpenCV Zoo).
# Outputs a 512-d embedding.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reid_model")
MODEL_PATH = os.path.join(MODEL_DIR, "person_reid_youtu_2021nov.onnx")

MATCH_THRESHOLD = 0.55

GALLERY_SIZE = 5

# Model's expected input, per OpenCV's official person_reid.py sample.
INPUT_WIDTH = 128
INPUT_HEIGHT = 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_lock = threading.Lock()
_net = None
_net_lock = threading.Lock()


def _get_net():
    """Lazily load the ONNX network once (cv2.dnn.Net load is expensive)."""
    global _net
    if _net is None:
        with _net_lock:
            if _net is None:
                if not os.path.exists(MODEL_PATH):
                    raise RuntimeError(
                        f"ReID model not found at {MODEL_PATH}. "
                        "Download person_reid_youtu_2021nov.onnx (see comment "
                        "at the top of reid_tracker.py) and place it there."
                    )
                net = cv2.dnn.readNetFromONNX(MODEL_PATH)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                _net = net
    return _net


def _load_db():
    """people_db.json - human-readable records (id, date, visits). No embeddings."""
    if not os.path.exists(REID_DB_PATH):
        return []
    with open(REID_DB_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_db(records):
    os.makedirs(REID_DIR, exist_ok=True)
    with open(REID_DB_PATH, "w") as f:
        json.dump(records, f, indent=2)


def _load_embeddings():
    #embeddings.json - {record_id (str): [embedding, embedding, ...]}.
    if not os.path.exists(EMBEDDINGS_PATH):
        return {}
    with open(EMBEDDINGS_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_embeddings(embeddings):
    os.makedirs(REID_DIR, exist_ok=True)
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(embeddings, f)


def _preprocess(crop_bgr):
    """Resize/normalize a person crop into the model's expected NCHW blob."""
    resized = cv2.resize(crop_bgr, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _MEAN) / _STD
    blob = normalized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
    return blob


def compute_embedding(crop_bgr):
    """
    Runs the pretrained ReID model on a person crop and returns an
    L2-normalized 512-d embedding (as a plain list, for JSON storage).
    Normalizing here means similarity is just a dot product later.
    """
    net = _get_net()
    blob = _preprocess(crop_bgr)
    net.setInput(blob)
    feat = net.forward().flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.tolist()


def _similarity(emb_a, emb_b):
    """Cosine similarity between two L2-normalized embeddings (dot product)."""
    a = np.array(emb_a, dtype=np.float32)
    b = np.array(emb_b, dtype=np.float32)
    return float(np.dot(a, b))


def process_entry(crop_bgr, image_path=None, timestamp=None):
    """
    Call this once per zone 'enter' event, passing the cropped person image
    (BGR, as produced by zone_counter._save_event) and the path where that
    crop was already saved on disk.

    Returns a dict:
      {
        "person_id": int,
        "is_new": bool,
        "visit_count": int,             
        "last_seen_before": dict|None,  
      }
    """
    now = timestamp or datetime.now()
    today_str = date.today().isoformat()
    embedding = compute_embedding(crop_bgr)
    visit_entry = {
        "timestamp": now.isoformat(sep=" ", timespec="seconds"),
        "image_path": image_path,
    }

    with _lock:
        records = _load_db()
        embeddings = _load_embeddings()
        today_records = [r for r in records if r["date"] == today_str]

        best_match = None
        best_score = -1.0
        for r in today_records:
            gallery = embeddings.get(str(r["id"]))
            if not gallery:
                continue
            score = max(_similarity(embedding, candidate) for candidate in gallery)
            if score > best_score:
                best_score = score
                best_match = r

        if best_match is not None and best_score >= MATCH_THRESHOLD:
            last_seen_before = best_match["visits"][-1]
            best_match["visits"].append(visit_entry)
            gallery = embeddings.setdefault(str(best_match["id"]), [])
            gallery.append(embedding)
            # FIFO cap: drop the oldest sample when GALLERY_SIZE is exceeded.
            if len(gallery) > GALLERY_SIZE:
                del gallery[: len(gallery) - GALLERY_SIZE]
            _save_db(records)
            _save_embeddings(embeddings)
            return {
                "person_id": best_match["id"],
                "is_new": False,
                "visit_count": len(best_match["visits"]),
                "last_seen_before": last_seen_before,
            }
        else:
            next_id = max((r["id"] for r in records), default=0) + 1
            new_record = {
                "id": next_id,
                "date": today_str,
                "visits": [visit_entry],
            }
            records.append(new_record)
            embeddings[str(next_id)] = [embedding]
            _save_db(records)
            _save_embeddings(embeddings)
            return {
                "person_id": next_id,
                "is_new": True,
                "visit_count": 1,
                "last_seen_before": None,
            }