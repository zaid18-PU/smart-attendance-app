# Smart Attendance AI

Click a photo of a class → instantly get who's present and who's absent, with an AI-generated summary report.

## How it works

```
Input Photo → Face Detection (RetinaFace) → Alignment → ViT Embedding (DINOv2)
            → Cosine Similarity Match vs Roster → Present/Absent Decision
            → SQLite Session Log → FLAN-T5 Generated Summary
```

Unlike classic pipelines that use a CNN (e.g. ArcFace) for face recognition, this project uses a **Vision Transformer** (`facebook/dinov2-base`, via HuggingFace `transformers`) to generate face embeddings, and a **second, language transformer** (`google/flan-t5-base`) to auto-write a natural-language attendance report — no manual report writing needed.

## Architecture

- **Backend**: FastAPI (Python) — REST API for enrollment, attendance, and history, backed by SQLite.
- **Frontend**: React (served as a single static bundle, no build step required — runs great in Colab).
- **Models**:
  - Detection: RetinaFace (`insightface`)
  - Recognition: DINOv2 ViT-Base (`transformers`)
  - Reporting: FLAN-T5-Base (`transformers`)

## Features

- Enroll students with reference photos (multiple photos per student supported)
- Upload a group photo → automatic face detection + matching + annotated result image
- Adjustable match-confidence threshold (live slider)
- Auto-generated natural-language attendance summary
- Full session history with per-student records
- Attendance-rate trend chart across sessions
- Clean, from-scratch designed dashboard UI (scanner/vision-inspired visual language)

## Running locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Then open `http://localhost:8000`.

## Running in Colab

Open `colab_run.ipynb` in Google Colab (GPU runtime), follow the cells top to bottom. It installs dependencies, launches the FastAPI server, and exposes it publicly via ngrok so you can demo it from any browser.

## Project structure

```
smart-attendance-app/
├── backend/
│   ├── main.py              # FastAPI app + API routes
│   ├── face_engine.py       # detection + ViT embedding + matching
│   ├── report_generator.py  # FLAN-T5 summary generation
│   ├── database.py          # SQLite persistence
│   └── requirements.txt
├── frontend/
│   └── static/index.html    # React app (CDN React, no bundler needed)
├── colab_run.ipynb          # one-click Colab launcher
└── README.md
```

## Evaluation

See the companion evaluation notebook (pipeline diagram, preprocessing visualization, confusion matrix, ROC-AUC, threshold sensitivity analysis on LFW benchmark pairs) for the technical report.

## Notes for capstone report

- **GPU utilization**: both the ViT recognition model and detector run on GPU; latency/throughput can be benchmarked and compared across T4 vs H200.
- **Transformer usage**: DINOv2 (vision transformer) for face recognition, FLAN-T5 (text transformer) for report generation — two distinct transformer applications in one pipeline.
- **Limitations**: recognition accuracy depends on photo quality/lighting/angle; current matching is a simple nearest-neighbor cosine threshold rather than a trained classifier head — a natural "future work" extension is fine-tuning a classification head on top of frozen DINOv2 features using enrolled student photos.
