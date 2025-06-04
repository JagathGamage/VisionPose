from fastapi import HTTPException 
import uuid
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

from pose.DPoseEstimationUsingYOLOv7 import process_and_dump ,generate_animation


os.chdir('..')
HOME1 = os.getcwd()
print("HOME1== ",HOME1)

# process_and_dump()
# Navigate from backend/app → backend → project-root → frontend/public/videos
SYNCED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend','frontend', 'public', 'videos'))
os.makedirs(SYNCED_DIR, exist_ok=True)


app = FastAPI()

# CORS Middleware (Allow React frontend to access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploaded_videos"
# SYNCED_DIR = "synced_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(SYNCED_DIR, exist_ok=True)

# Directory to save projects and serve static files
PROJECTS_DIR = "projects"
STATIC_FILES_DIR = "C:/Users/ucsc/Desktop/vs/VisionPose/frontend/frontend/public/videos"  # Specify your frontend folder here

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
async def sync_videos(
    file_paths: List[UploadFile] = File(...),
    sync_times: List[float] = Form(...)
):
    if len(file_paths) != 3 or len(sync_times) != 3:
        return {"message": "Exactly 3 videos and 3 sync times are required."}

    temp_video_paths = []

    # Save uploaded videos to disk
    for file in file_paths:
        temp_path = f"temp_{uuid.uuid4().hex}.mp4"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_video_paths.append(temp_path)

    try:
        min_sync_time = min(sync_times)
        synced_paths = []

        for i, (path, sync_time) in enumerate(zip(temp_video_paths, sync_times)):
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            output_filename = f"synced_{i+1}.mp4"
            output_path = os.path.join(SYNCED_DIR, output_filename)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            start_frame = int((sync_time - min_sync_time) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()
            out.release()
            synced_paths.append(f"/videos/{output_filename}")  # Relative path for frontend
            print("hello jaga")

        return {"message": "Videos synced successfully.", "output_paths": synced_paths}
        

    finally:
        for path in temp_video_paths:
            os.remove(path)
        print("finally")


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
TRIM_DIR = os.path.join( "pose", "input")
# TRIM_DIR = "trimed_videos"
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
    

SOURCE_VIDEO_A_PATH = f"{HOME1}/pose/input/vid1.mp4"
SOURCE_VIDEO_B_PATH = f"{HOME1}/pose/input/vid1.mp4"
SOURCE_VIDEO_C_PATH = f"{HOME1}/pose/input/vid1.mp4"
    
@app.post("/processAndDump/")
async def processAndDump():
    process_and_dump(SOURCE_VIDEO_A_PATH, f"{HOME1}/pose/output/pose-estimation-synchronised-sample-a.json", f"{HOME1}/pose/output/right-shoulder-angle-a.xlsx", f"{HOME1}/pose/output/right-shoulder-angles-sample-a.mp4", f"{HOME1}/pose/output/right-shoulder-angles-sample-a.png", f"{HOME1}/pose/output")
    print("hello")
    process_and_dump(SOURCE_VIDEO_B_PATH, f"{HOME1}/pose/output/pose-estimation-synchronised-sample-b.json", f"{HOME1}/pose/output/right-shoulder-angle-b.xlsx", f"{HOME1}/pose/output/right-shoulder-angles-sample-b.mp4", f"{HOME1}/pose/output/right-shoulder-angles-sample-b.png", f"{HOME1}/pose/output")
    print("hello")
    process_and_dump(SOURCE_VIDEO_C_PATH, f"{HOME1}/pose/output/pose-estimation-synchronised-sample-c.json", f"{HOME1}/pose/output/right-shoulder-angle-c.xlsx", f"{HOME1}/pose/output/right-shoulder-angles-sample-c.mp4", f"{HOME1}/pose/output/right-shoulder-angles-sample-c.png", f"{HOME1}/pose/output")
    print("hello")

@app.post("/animation/")
async def animation():
    print("HOME1 in ani",HOME1)
    generate_animation()
    return "success anime"
