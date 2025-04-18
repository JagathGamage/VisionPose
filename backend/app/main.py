from http.client import HTTPException
from fastapi import FastAPI, File, UploadFile, Form
import os
import cv2
import numpy as np
from typing import List
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pose.DPoseEstimationUsingYOLOv7 import process_and_dump

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
async def upload_videos(
    project_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    if len(files) != 3:
        return {"error": "Exactly 3 video files are required"}

    project_path = os.path.join(PROJECTS_DIR, project_name)
    os.makedirs(project_path, exist_ok=True)

    video_paths = []
    for file in files:
        video_path = os.path.join(project_path, file.filename)
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())
        video_paths.append(video_path)

    # Sync videos
    synced_paths = sync_videos(video_paths, project_path)
    
    # Move synced videos to frontend static folder
    for path in synced_paths:
        shutil.move(path, os.path.join(STATIC_FILES_DIR, os.path.basename(path)))

    return {"message": "Videos uploaded, synced, and moved to frontend folder", "synced_files": synced_paths}

def sync_videos(video_paths, project_path):
    """Syncs videos based on the first common frame"""
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    first_frames = []

    for cap in caps:
        ret, frame = cap.read()
        if ret:
            first_frames.append(frame)

    ref_frame = first_frames[0]
    
    # Finding sync points (first matching frame)
    sync_points = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if np.array_equal(frame, ref_frame):
                sync_points.append(frame_index)
                break
            frame_index += 1
        cap.release()

    min_sync = min(sync_points)

    # Trim videos to sync
    synced_paths = []
    for idx, vp in enumerate(video_paths):
        trimmed_path = os.path.join(project_path, f"synced_{idx+1}.mp4")
        trim_video(vp, trimmed_path, min_sync)
        synced_paths.append(trimmed_path)

    return synced_paths

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
UPLOAD_DIR = "backend_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/uploadTrimedVideos/")
async def upload_trimed_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Video '{file.filename}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/processAndDump/")
async def processAndDump():
    # process_and_dump()
    print("hello")