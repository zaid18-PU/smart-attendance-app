"""
Smart Attendance AI — FastAPI backend.

Pipeline: RetinaFace detection -> DINOv2 (ViT) embedding -> cosine matching
against roster -> SQLite persistence -> FLAN-T5 generated summary.

Run locally:
    uvicorn main:app --reload --port 8000

Run in Colab: see colab_run.ipynb in the project root.
"""
import base64
import io
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import database as db
import face_engine
import report_generator

app = FastAPI(title="Smart Attendance AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MATCH_THRESHOLD_DEFAULT = 0.55  # DINOv2 cosine similarity threshold (tune via evaluation notebook)


@app.on_event("startup")
def startup():
    db.init_db()
    face_engine.load_models()
    report_generator.load_report_model()
    print("[main] Smart Attendance AI backend ready.")


def read_upload_as_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def encode_bgr_to_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Roster endpoints
# ---------------------------------------------------------------------------

@app.get("/api/roster")
def api_get_roster():
    return {"roster": db.get_roster()}


@app.post("/api/enroll")
async def api_enroll(name: str = Form(...), file: UploadFile = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")

    img_bgr = read_upload_as_bgr(await file.read())
    faces = face_engine.detect_faces(img_bgr)

    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No face detected in the photo.")
    if len(faces) > 1:
        raise HTTPException(status_code=422, detail="Multiple faces detected — upload a photo with exactly one face.")

    vec = face_engine.embed_face(img_bgr, faces[0].bbox)
    student_id = db.get_or_create_student(name)
    db.add_embedding(student_id, vec.tolist())

    return {"message": f"Enrolled a reference photo for {name}.", "student_id": student_id}


@app.delete("/api/roster/{student_id}")
def api_delete_student(student_id: int):
    db.delete_student(student_id)
    return {"message": "Deleted."}


@app.delete("/api/roster")
def api_clear_roster():
    db.clear_roster()
    return {"message": "Roster cleared."}


# ---------------------------------------------------------------------------
# Attendance endpoint
# ---------------------------------------------------------------------------

class AttendanceResponse(BaseModel):
    annotated_image: str
    present: list
    absent: list
    records: list
    summary: str
    session_id: int


@app.post("/api/attendance")
async def api_take_attendance(file: UploadFile = File(...), threshold: float = Form(MATCH_THRESHOLD_DEFAULT)):
    roster = db.get_roster()
    if not roster:
        raise HTTPException(status_code=400, detail="Roster is empty — enroll students first.")

    roster_embeddings = db.get_all_embeddings()
    img_bgr = read_upload_as_bgr(await file.read())
    faces = face_engine.detect_faces(img_bgr)

    matched_ids = set()
    draw_results = []
    records = []

    for face in faces:
        vec = face_engine.embed_face(img_bgr, face.bbox)
        sid, name, score = face_engine.match_embedding(vec, roster_embeddings, threshold)
        matched = sid is not None
        if matched:
            matched_ids.add(sid)
        draw_results.append({
            "bbox": face.bbox.tolist(),
            "label": name,
            "score": score,
            "matched": matched,
        })

    all_students = {r["id"]: r["name"] for r in roster}
    present_names, absent_names = [], []

    for sid, name in all_students.items():
        if sid in matched_ids:
            present_names.append(name)
        else:
            absent_names.append(name)

    for sid in matched_ids:
        best_score = max(
            (dr["score"] for dr in draw_results if dr["label"] == all_students.get(sid)),
            default=None,
        )
        records.append({"student_id": sid, "name": all_students[sid], "status": "Present", "confidence": best_score})
    for name in absent_names:
        records.append({"student_id": None, "name": name, "status": "Absent", "confidence": None})

    summary = report_generator.generate_summary(present_names, absent_names, len(roster))

    session_id = db.create_session(
        total_roster=len(roster),
        total_present=len(present_names),
        total_absent=len(absent_names),
        summary=summary,
        records=records,
    )

    annotated = face_engine.draw_annotations(img_bgr, draw_results)
    annotated_b64 = encode_bgr_to_b64(annotated)

    records_sorted = sorted(records, key=lambda r: (r["status"] != "Present", r["name"]))

    return AttendanceResponse(
        annotated_image=annotated_b64,
        present=sorted(present_names),
        absent=sorted(absent_names),
        records=records_sorted,
        summary=summary,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# History / analytics endpoints
# ---------------------------------------------------------------------------

@app.get("/api/history")
def api_history(limit: int = 50):
    return {"history": db.get_history(limit)}


@app.get("/api/history/{session_id}")
def api_history_detail(session_id: int):
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Session not found.")
    return detail


@app.get("/api/health")
def api_health():
    return {"status": "ok", "time": time.time()}


# ---------------------------------------------------------------------------
# Serve the React frontend (static build) at the root
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="../frontend/static", html=True), name="static")
