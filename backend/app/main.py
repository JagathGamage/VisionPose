from http.client import HTTPException
from fastapi import FastAPI, File, UploadFile, Form
import os
import cv2
import numpy as np
from typing import List
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from uuid import uuid4
# from pose.DPoseEstimationUsingYOLOv7 import process_and_dump

# process_and_dump()


app = FastAPI()

# CORS Middleware (Allow React frontend to access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploaded_videos"
SYNCED_DIR = "synced_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SYNCED_DIR, exist_ok=True)

# Directory to save projects and serve static files
PROJECTS_DIR = "projects"
STATIC_FILES_DIR = "C:/Users/Jagath/Desktop/VisionPose/VisionPose/frontend/frontend/src/videos"  # Specify your frontend folder here

# Ensure directories exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(STATIC_FILES_DIR, exist_ok=True)

# Serve video files from the frontend's public folder
app.mount("/videos", StaticFiles(directory=STATIC_FILES_DIR), name="videos")

@app.get("/")
async def hello():
    return "hello"

@app.post("/upload/")
async def upload_videos(project_name: str = Form(...), files: List[UploadFile] = File(...)):
    saved_files = []

    for file in files:
        filename = f"{uuid4()}_{file.filename}"
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        saved_files.append(path)

    return JSONResponse({
        "message": "Videos uploaded successfully",
        "saved_files": saved_files
    })


@app.post("/sync/")
async def sync_videos(file_paths: List[UploadFile] = File(...), sync_times: List[float] = Form(...)):
    synced_paths = []

    for idx, file in enumerate(file_paths):
        filename = f"{uuid4()}_{file.filename}"
        video_path = os.path.join(UPLOAD_DIR, filename)
        with open(video_path, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(sync_times[idx] * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if success:
            out_path = os.path.join(SYNCED_DIR, f"synced_{idx}_{file.filename}")
            height, width, _ = frame.shape
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            out.write(frame)
            out.release()
            synced_paths.append(out_path)
        cap.release()

    return JSONResponse({
        "message": "Videos synced successfully by selected frames.",
        "synced_files": synced_paths
    })

def trim_video(input_path, output_path, start_frame):
    """Trims the video starting from the given frame"""
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use H264 codec for widely supported videos
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

# Directory to save videos
TRIM_DIR = "trimed_videos"
os.makedirs(TRIM_DIR, exist_ok=True)

@app.post("/uploadTrimedVideos/")
async def upload_trimed_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(TRIM_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Video '{file.filename}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/processAndDump/")
async def processAndDump():
    # process_and_dump()
    print("hello")