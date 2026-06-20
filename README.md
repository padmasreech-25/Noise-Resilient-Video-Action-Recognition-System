# Noise-Resilient Video Action Recognition System
**Stack**: React (frontend) + FastAPI + PyTorch SRGAN + InceptionV3 (backend)

---

## Project Structure
```
project/
├── backend/
│   ├── main.py              ← FastAPI app (SRGAN + InceptionV3)
│   └── requirements.txt
└── frontend/
    └── src/
        └── App.jsx          ← React UI
```

---

## Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### Optional: Pretrained Weights
- Place SRGAN weights as `backend/srgan_weights.pth`
- Place InceptionV3 weights as `backend/inception_weights.pth`
- Without weights, the models run with random initialization (for demo/testing)
- For real results, fine-tune InceptionV3 on UCF-101 dataset

---

## Frontend Setup

```bash
cd frontend

# Install (create React app or Vite)
npm create vite@latest . -- --template react
npm install

# Replace src/App.jsx with the provided App.jsx
# Then run:
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload-and-process` | Upload video, start processing |
| GET | `/job/{job_id}` | Poll job status + results |
| GET | `/download/{job_id}` | Download enhanced video |
| DELETE | `/job/{job_id}` | Clean up job |
| GET | `/health` | Check GPU/CPU status |

---

## How It Works

1. **User uploads** blurry/low-quality video via React UI
2. **FastAPI** receives it and starts a background job
3. **SRGAN Generator** processes each frame → 4× upscaled enhanced frames
4. **InceptionV3** samples frames at regular intervals → predicts action class
5. **Enhanced video** is written as MP4 and made available for download
6. **React UI** polls for job completion, displays top-5 action predictions with confidence scores, and offers download button

---

## Training Notes (for production)
- **SRGAN**: Train on DIV2K dataset (high-res → artificially degraded pairs)
- **InceptionV3**: Fine-tune on UCF-101 (101-class action recognition dataset)
  - Download: https://www.crcv.ucf.edu/data/UCF101.php
  - Replace `len(ACTION_LABELS)` with actual number of classes used
