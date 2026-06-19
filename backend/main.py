"""
Noise-Resilient Video Action Recognition System
- High quality video enhancement (Bilateral + CLAHE + Unsharp masking)
- Optical Flow motion analysis for accurate action detection
- InceptionV3 scene context with sport/activity boosting
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os, uuid, cv2, numpy as np, torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
import shutil
from pathlib import Path
from typing import List, Dict
import time, traceback, warnings, threading
warnings.filterwarnings("ignore")

app = FastAPI(title="Noise-Resilient Video Action Recognition", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ACTION_LABELS = [
    "Sitting",      "Standing",     "Walking",      "Running",       "Jumping",
    "Dancing",      "Waving",       "Clapping",     "Stretching",    "Exercising",
    "Basketball",   "Tennis",       "Golf",         "Swimming",      "Cycling",
    "Push Ups",     "Squats",       "Lunges",       "Pull Ups",      "Boxing",
    "Cooking",      "Eating",       "Drinking",     "Reading",       "Talking",
    "Typing",       "Writing",      "Singing",      "Playing Music", "Yoga",
    "Climbing",     "Diving",       "Surfing",      "Skiing",        "Skateboarding",
    "Horse Riding", "Rowing",       "Sky Diving",   "Gymnastics",    "Martial Arts"
]

IMAGENET_TO_ACTION = {
    "dumbbell":"Exercising",      "barbell":"Exercising",       "basketball":"Basketball",
    "tennis_ball":"Tennis",       "tennis_racket":"Tennis",     "golf_ball":"Golf",
    "golf":"Golf",                "soccer_ball":"Running",      "surfboard":"Surfing",
    "ski":"Skiing",               "snowboard":"Skiing",         "bicycle":"Cycling",
    "mountain_bike":"Cycling",    "skateboard":"Skateboarding", "horse":"Horse Riding",
    "canoe":"Rowing",             "kayak":"Rowing",             "parachute":"Sky Diving",
    "punching_bag":"Boxing",      "boxing_glove":"Boxing",      "guitar":"Playing Music",
    "piano":"Playing Music",      "violin":"Playing Music",     "drum":"Playing Music",
    "microphone":"Singing",       "desk":"Sitting",             "chair":"Sitting",
    "armchair":"Sitting",         "sofa":"Sitting",             "couch":"Sitting",
    "barbershop":"Sitting",       "barber_chair":"Sitting",     "restaurant":"Eating",
    "dining_table":"Eating",      "kitchen":"Cooking",          "stove":"Cooking",
    "fork":"Eating",              "spoon":"Eating",             "cup":"Drinking",
    "wine_glass":"Drinking",      "coffee_mug":"Drinking",      "book":"Reading",
    "bookshelf":"Reading",        "laptop":"Typing",            "computer_keyboard":"Typing",
    "notebook":"Writing",         "pencil":"Writing",           "classroom":"Sitting",
    "office":"Sitting",           "living_room":"Sitting",      "bookshop":"Reading",
    "trampoline":"Jumping",       "horizontal_bar":"Pull Ups",  "rope":"Climbing",
    "running_shoe":"Running",     "mobile_home":"Sitting",      "bed":"Sitting",
    "pillow":"Sitting",           "lounge":"Sitting",           "cowboy_boot":"Standing",
    "sandbar":"Sitting",          "seashore":"Sitting",         "swimming_trunks":"Swimming",
    "scuba_diver":"Diving",       "volleyball":"Exercising",    "football_helmet":"Running",
    "swimming_cap":"Swimming",    "balance_beam":"Gymnastics",  "parallel_bars":"Gymnastics",
    "weight":"Exercising",        "yoga_mat":"Yoga",            "ping-pong":"Tennis",
}

# ── InceptionV3 ───────────────────────────────────────────────────────────────
inception_model = None
imagenet_labels = None

def get_inception():
    global inception_model, imagenet_labels
    if inception_model is None:
        print("Loading InceptionV3 with ImageNet weights...")
        weights = Inception_V3_Weights.IMAGENET1K_V1
        inception_model = inception_v3(weights=weights)
        inception_model.aux_logits = False
        inception_model.AuxLogits = None
        inception_model = inception_model.to(device)
        inception_model.eval()
        imagenet_labels = weights.meta["categories"]
        print("InceptionV3 ready.")
    return inception_model, imagenet_labels

action_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── HIGH QUALITY Frame Enhancement ───────────────────────────────────────────
def enhance_frame_hq(frame: np.ndarray, scale: int = 2) -> np.ndarray:
    # Step 1: Bilateral denoise — removes noise, keeps edges sharp
    denoised = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: CLAHE contrast enhancement on luminance
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)

    # Step 3: Bicubic 2x upscale
    h, w = contrast.shape[:2]
    upscaled = cv2.resize(contrast, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Step 4: Unsharp masking for crisp details
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2.0)
    unsharp = cv2.addWeighted(upscaled, 1.6, gaussian, -0.6, 0)

    # Step 5: Saturation boost for vivid colors
    hsv = cv2.cvtColor(unsharp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Step 6: Final soft sharpening pass
    blend_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
    result = cv2.filter2D(saturated, -1, blend_kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


# ── OPTICAL FLOW Motion Analysis ──────────────────────────────────────────────
def analyze_motion(gray_frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Measures actual pixel movement between frames using optical flow.
    Works correctly even on blurry/low-quality input videos.
    """
    if len(gray_frames) < 2:
        return {"Sitting": 1.0}

    magnitudes = []
    for i in range(1, min(len(gray_frames), 20)):
        try:
            f1 = cv2.equalizeHist(gray_frames[i-1])
            f2 = cv2.equalizeHist(gray_frames[i])
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            magnitudes.append(float(mag))
        except:
            continue

    if not magnitudes:
        return {"Sitting": 1.0}

    avg_mag    = float(np.mean(magnitudes))
    std_mag    = float(np.std(magnitudes))
    max_mag    = float(np.max(magnitudes))
    # Combine avg + burst for better sensitivity
    motion_level = max(avg_mag, max_mag * 0.3)

    print(f"Motion: avg={avg_mag:.4f} std={std_mag:.4f} max={max_mag:.4f} level={motion_level:.4f}")

    if motion_level < 0.3:
        return {"Sitting": 0.75, "Talking": 0.15, "Standing": 0.10}

    elif motion_level < 0.8:
        return {"Sitting": 0.45, "Talking": 0.25, "Standing": 0.20, "Waving": 0.10}

    elif motion_level < 1.5:
        return {"Standing": 0.35, "Walking": 0.30, "Waving": 0.20, "Clapping": 0.15}

    elif motion_level < 3.0:
        return {"Walking": 0.50, "Standing": 0.20, "Waving": 0.15, "Stretching": 0.15}

    elif motion_level < 6.0:
        if std_mag > 1.5:
            return {"Dancing": 0.35, "Exercising": 0.30, "Running": 0.20, "Jumping": 0.15}
        return {"Running": 0.40, "Walking": 0.25, "Exercising": 0.20, "Stretching": 0.15}

    elif motion_level < 10.0:
        return {"Running": 0.30, "Jumping": 0.25, "Boxing": 0.20, "Dancing": 0.15, "Martial Arts": 0.10}

    else:
        return {"Gymnastics": 0.30, "Martial Arts": 0.25, "Jumping": 0.20, "Boxing": 0.15, "Sky Diving": 0.10}


# ── InceptionV3 Scene Context ─────────────────────────────────────────────────
def get_inception_scores(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Detects objects/scene in video frames.
    Sports equipment gets 3x boost — dominates over sitting/standing.
    """
    scores: Dict[str, float] = {a: 0.0 for a in ACTION_LABELS}
    # Static/passive actions — low boost
    passive = {"Sitting", "Standing", "Talking", "Reading", "Typing", "Writing"}
    try:
        model, labels = get_inception()
        step = max(1, len(frames) // 8)
        sample = frames[::step][:8]
        for frame in sample:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = action_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1).squeeze(0).cpu()
            top10 = torch.topk(probs, 10)
            for score, idx in zip(top10.values.tolist(), top10.indices.tolist()):
                label = labels[idx].lower().replace(" ", "_").replace(",", "")
                for keyword, action in IMAGENET_TO_ACTION.items():
                    if keyword in label and action in scores:
                        # Sports/active actions get 3x boost over passive ones
                        boost = 1.0 if action in passive else 3.0
                        scores[action] += float(score) * boost
                        break
    except Exception as e:
        print(f"InceptionV3 error: {e}")
    return scores


# ── MAIN VIDEO PIPELINE ───────────────────────────────────────────────────────
def process_video(input_path: str, output_path: str, sample_rate: int = 8):
    print(f"\n{'='*55}")
    print(f"Processing: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input:  {width}x{height} @ {fps:.1f}fps | {total} frames")

    scale = 2
    out_w, out_h = width * scale, height * scale
    print(f"Output: {out_w}x{out_h} (2x high quality enhancement)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        raise ValueError("Cannot create output writer")

    enhanced_frames: List[np.ndarray] = []
    gray_frames:     List[np.ndarray] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = enhance_frame_hq(frame, scale=scale)
        out.write(enhanced)

        if idx % sample_rate == 0:
            enhanced_frames.append(enhanced.copy())
            gray = cv2.cvtColor(
                cv2.resize(enhanced, (width, height)),
                cv2.COLOR_BGR2GRAY
            )
            gray_frames.append(gray)

        idx += 1
        if idx % 50 == 0:
            print(f"  Enhanced {idx}/{total} frames...")

    cap.release()
    out.release()
    print(f"Enhancement complete: {idx} frames → {out_w}x{out_h}")

    if not enhanced_frames:
        raise ValueError("No frames processed")

    # Motion analysis (primary)
    print("Analyzing motion with optical flow...")
    motion_scores = analyze_motion(gray_frames)

    # Scene/object detection (secondary)
    print("Running InceptionV3 scene analysis...")
    inception_scores = get_inception_scores(enhanced_frames)

    # Combine: 50% motion + 50% scene
    # If InceptionV3 detects strong sports object → it will dominate
    combined: Dict[str, float] = {}
    for action in ACTION_LABELS:
        m = motion_scores.get(action, 0.0)
        v = inception_scores.get(action, 0.0)
        combined[action] = (m * 0.50) + (v * 0.50)

    sorted_actions = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    total_score    = sum(s for _, s in sorted_actions[:10]) or 1.0

    top5 = [
        {"action": action, "confidence": round(score / total_score, 4)}
        for action, score in sorted_actions[:5]
        if score > 0
    ]

    fallback = ["Sitting", "Standing", "Walking", "Running", "Talking"]
    while len(top5) < 5:
        top5.append({"action": fallback[len(top5)], "confidence": 0.01})

    top5_display = [(t["action"], f"{t['confidence']*100:.1f}%") for t in top5]
    print(f"Top action: {top5[0]['action']} ({top5[0]['confidence']*100:.1f}%)")
    print(f"Top 5: {top5_display}")
    print("="*55)

    return {
        "top_action":          top5[0]["action"],
        "confidence":          top5[0]["confidence"],
        "top5":                top5,
        "total_frames":        idx,
        "enhanced_resolution": f"{out_w}x{out_h}",
        "original_resolution": f"{width}x{height}",
    }


# ── Job Store ─────────────────────────────────────────────────────────────────
jobs: Dict[str, Dict] = {}

def run_processing_job(job_id: str, input_path: str, output_path: str):
    jobs[job_id]["status"] = "processing"
    try:
        result = process_video(input_path, output_path)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        print(f"Job {job_id} DONE.")
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Noise-Resilient Video Action Recognition API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "cuda_available": torch.cuda.is_available()}

@app.post("/upload-and-process")
async def upload_and_process(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported format.")
    job_id      = str(uuid.uuid4())
    input_path  = str(UPLOAD_DIR / f"{job_id}_input.mp4")
    output_path = str(OUTPUT_DIR / f"{job_id}_enhanced.mp4")
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
    print(f"Saved {input_path} ({len(contents)} bytes)")
    jobs[job_id] = {
        "status": "queued", "input_path": input_path, "output_path": output_path,
        "filename": file.filename, "created_at": time.time(), "result": None, "error": None,
    }
    t = threading.Thread(target=run_processing_job, args=(job_id, input_path, output_path), daemon=True)
    t.start()
    print(f"Thread started for job {job_id}")
    return {"job_id": job_id, "status": "queued", "message": "Processing started"}

@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {"job_id": job_id, "status": job["status"], "filename": job["filename"],
            "result": job["result"], "error": job["error"]}

@app.get("/download/{job_id}")
def download_enhanced_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Not ready: {job['status']}")
    if not Path(job["output_path"]).exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(job["output_path"], media_type="video/mp4",
                        filename=f"enhanced_{job['filename']}")

@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs.pop(job_id)
    for path in [job.get("input_path"), job.get("output_path")]:
        if path and Path(path).exists():
            os.remove(path)
    return {"message": "Job deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)