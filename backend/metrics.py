"""
Model performance evaluation for the DINOv2 face verification pipeline.

Uses pairwise comparisons between embeddings already stored in the roster
(no external labeled dataset needed) -- since every embedding is tagged with
a known student ID, every pair of embeddings is either a genuine "same
student" pair or an impostor "different student" pair. This is a standard
technique for evaluating face verification systems.

Note: students need 2+ reference photos each to produce genuine (positive)
pairs. With only 1 photo per student you'll still get negative-pair
separability, but no positive pairs / recall.
"""
import numpy as np
import database as db
import face_engine


def evaluate_verification(threshold=0.55):
    data = db.get_all_embeddings()  # {student_id: {name, vectors: [...]}}
    items = [(sid, v) for sid, d in data.items() for v in d["vectors"]]

    pairs = []  # (score, label) label=1 same student, 0 different
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sid_i, vec_i = items[i]
            sid_j, vec_j = items[j]
            score = face_engine.cosine_sim(np.array(vec_i, dtype=np.float32), np.array(vec_j, dtype=np.float32))
            label = 1 if sid_i == sid_j else 0
            pairs.append((score, label))

    if not pairs:
        return {
            "error": "Not enough embeddings to evaluate. Enroll at least 2 students "
                     "(ideally with 2+ photos each) before running evaluation.",
            "num_pairs": 0,
        }

    scores = np.array([p[0] for p in pairs])
    labels = np.array([p[1] for p in pairs])
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(pairs)

    # ROC-AUC via rank statistic (no sklearn dependency)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    auc = float(np.mean([np.mean(p > neg) for p in pos])) if len(pos) and len(neg) else None

    return {
        "threshold": threshold,
        "num_pairs": len(pairs),
        "num_positive_pairs": int((labels == 1).sum()),
        "num_negative_pairs": int((labels == 0).sum()),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "score_distribution": {
            "positive_pairs_mean_score": round(float(pos.mean()), 4) if len(pos) else None,
            "negative_pairs_mean_score": round(float(neg.mean()), 4) if len(neg) else None,
        },
    }
