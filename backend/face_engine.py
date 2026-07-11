"""
Face engine: detection (RetinaFace via insightface) + recognition using a
Vision Transformer (DINOv2, via HuggingFace `transformers`) instead of a
classic CNN embedder. This is the "use of transformers" component of the
pipeline: DINOv2 is a self-supervised ViT that produces strong general-purpose
visual embeddings, which we repurpose here for face verification.
"""
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from insightface.app import FaceAnalysis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_detector = None
_vit_processor = None
_vit_model = None


def load_models():
    global _detector, _vit_processor, _vit_model

    print(f"[face_engine] Loading detector (RetinaFace) on {DEVICE} ...")
    _detector = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ctx_id = 0 if DEVICE == "cuda" else -1
    _detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

    print(f"[face_engine] Loading ViT recognizer (facebook/dinov2-base) on {DEVICE} ...")
    _vit_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _vit_model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()

    print("[face_engine] Models ready.")


def detect_faces(img_bgr: np.ndarray):
    """Returns list of insightface Face objects (has .bbox, .kps)."""
    return _detector.get(img_bgr)


def _crop_face(img_bgr: np.ndarray, bbox, margin=0.2):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))
    return img_bgr[y1:y2, x1:x2]


@torch.no_grad()
def embed_face(img_bgr: np.ndarray, bbox) -> np.ndarray:
    """Crop the face region and produce a normalized ViT embedding vector."""
    crop_bgr = _crop_face(img_bgr, bbox)
    if crop_bgr.size == 0:
        crop_bgr = img_bgr
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    inputs = _vit_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    outputs = _vit_model(**inputs)
    # CLS token as the global face representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    vec = cls_embedding.cpu().numpy()
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def match_embedding(query_vec: np.ndarray, roster_embeddings: dict, threshold: float):
    """
    roster_embeddings: {student_id: {"name": str, "vectors": [list, ...]}}
    Returns (student_id_or_None, name_or_'Unknown', best_score)
    """
    best_id, best_name, best_score = None, "Unknown", -1.0
    for sid, data in roster_embeddings.items():
        for v in data["vectors"]:
            v = np.array(v, dtype=np.float32)
            score = cosine_sim(query_vec, v)
            if score > best_score:
                best_score = score
                best_id = sid
                best_name = data["name"]
    if best_score < threshold:
        return None, "Unknown", best_score
    return best_id, best_name, best_score


def draw_annotations(img_bgr: np.ndarray, results: list) -> np.ndarray:
    """results: list of dicts with bbox, label, matched (bool)"""
    annotated = img_bgr.copy()
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        color = (0, 200, 0) if r["matched"] else (0, 0, 230)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{r['label']} ({r['score']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return annotated
