# backend/main.py
import base64
import os
from pathlib import Path
import shutil
from typing import List
import uuid
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from uuid import uuid4
import subprocess


#from pose.DPoseEstimationUsingYOLOv7 import process_and_dump ,generate_animation


os.chdir('..')
HOME1 = os.getcwd()
print("HOME1== ",HOME1)

# Where “synced_*.mp4” will be written (also served via the frontend's /videos route):
SYNCED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "frontend", "public", "videos")
)
os.makedirs(SYNCED_DIR, exist_ok=True)

app = FastAPI()

# Allow React (http://localhost:3000) or any origin to talk to this API:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FFMPEG_PATH = r"C:\Users\Jagath\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# This folder is only used for temporarily storing uploads before we process them:
UPLOAD_DIR = HOME1
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory to save projects and serve static files
PROJECTS_DIR = "projects"
#STATIC_FILES_DIR = "C:/Users/ucsc/Desktop/vs/VisionPose/frontend/frontend/public/videos"  # Specify your frontend folder here
STATIC_FILES_DIR="C:/Users/Jagath/Desktop/VisionPose/VisionPose/frontend/frontend/public/videos"
# Ensure directories exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(STATIC_FILES_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=STATIC_FILES_DIR), name="videos")
app.mount("/output", StaticFiles(directory="C:/Users/Jagath/Desktop/VisionPose/VisionPose/backend/pose/output"), name="output")
app.mount("/animation", StaticFiles(directory="C:/Users/Jagath/Desktop/VisionPose/VisionPose/backend"), name="animation")
app.mount("/synced_videos", StaticFiles(directory="C:/Users/Jagath/Desktop/VisionPose/VisionPose/backend/frontend/frontend/public/videos"), name="synced_videos")
app.mount("/uploaded_videos", StaticFiles(directory="C:/Users/Jagath/Desktop/VisionPose/VisionPose/backend/uploaded_videos"), name="uploaded_videos")


@app.get("/")
async def hello():
    return {"message": "hello from backend"}

# UPLOAD_DIR = "uploads"
 #project_dir = os.path.join(UPLOAD_DIR, "backend","uploaded_videos")
@app.get("/extract_frames/{video_name}")
def extract_frames(video_name: str, fps_extract: int = 10):  # default to 5 fps extraction
    video_path = os.path.join(UPLOAD_DIR, "backend","uploaded_videos", video_name)

    if not os.path.isfile(video_path):
        return JSONResponse(status_code=404, content={"detail": "Video not found"})

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 25  # fallback fps if unavailable

    # Calculate frame interval to get approx 'fps_extract' frames per second
    frame_interval = max(int(video_fps / fps_extract), 1)

    frames = []
    frame_count = 0
    success, image = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode('.jpg', image)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            time_sec = frame_count / video_fps
            frames.append({
                "time": round(time_sec, 3),
                "image": f"data:image/jpeg;base64,{jpg_as_text}"
            })
        success, image = cap.read()
        frame_count += 1

    cap.release()

    return {"frames": frames}
@app.post("/upload/")
async def upload_videos(
    project_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if len(files) != 3:
        return {"error": "Exactly 3 video files are required."}

    # Create project-specific directory
    project_dir = os.path.join(UPLOAD_DIR, "backend","uploaded_videos")
    os.makedirs(project_dir, exist_ok=True)

    reencoded_files = []

    # Save and re-encode each video
    for i, file in enumerate(files):
        file_ext = os.path.splitext(file.filename)[1]
        original_path = os.path.join(project_dir, f"original_video_{i+1}{file_ext}")
        reencoded_path = os.path.join(project_dir, f"video_{i+1}_formatted.mp4")

        # Save original file
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Re-encode with FFmpeg
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", original_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart", reencoded_path
        ])

        # Optionally delete the original
        #os.remove(original_path)

        reencoded_files.append(reencoded_path)

    return {
        "message": "Videos uploaded and re-encoded successfully",
        "project": project_name,
        "files": reencoded_files
    }

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

            # FFmpeg output re-encoded path
            reencoded_path = os.path.join(SYNCED_DIR, f"synced_{i+1}_web.mp4")

            # Re-encode using FFmpeg to H.264 with faststart
            subprocess.run([
                r"C:\Users\Jagath\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart", reencoded_path
            ])

            # Optionally delete the original non-browser-friendly video
            os.remove(output_path)

            # Save the new browser-compatible path
            synced_paths.append(f"/videos/synced_{i+1}_web.mp4")

            print("hello jaga")

        return {"message": "Videos synced successfully.", "output_paths": synced_paths}
        

    finally:
        for path in temp_video_paths:
            os.remove(path)
        print("finally")

# ----------------------------
# (Existing trim‐upload endpoints follow—unchanged)
# ----------------------------

TRIM_DIR = os.path.join("C:/Users/Jagath/Desktop/VisionPose/VisionPose/backend", "pose", "input")
os.makedirs(TRIM_DIR, exist_ok=True)


@app.post("/uploadTrimedVideos/")
async def upload_trimed_video(file: UploadFile = File(...)):
    try:
        dest = os.path.join(TRIM_DIR, file.filename)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Video '{file.filename}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

SOURCE_VIDEO_A_PATH = f"{HOME1}/pose/input/trimmed0.mp4"
SOURCE_VIDEO_B_PATH = f"{HOME1}/pose/input/trimmed1.mp4"
SOURCE_VIDEO_C_PATH = f"{HOME1}/pose/input/trimmed2.mp4"

FFMPEG_PATH = r"C:\Users\Jagath\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
OUTPUT_DIRECTROY = Path(r"C:\Users\Jagath\Desktop\VisionPose\VisionPose\backend\pose\output")

    
@app.post("/processAndDump/")
async def processAndDump():
    #process_and_dump(SOURCE_VIDEO_A_PATH, f"{HOME1}/pose/output/pose-estimation-synchronised-sample-a.json", f"{HOME1}/pose/output/right-shoulder-angle-a.xlsx", f"{HOME1}/pose/output/right-shoulder-angles-sample-a.mp4", f"{HOME1}/pose/output/right-shoulder-angles-sample-a.png", f"{HOME1}/pose/output")
    print("hello process and dump")
   #process_and_dump(SOURCE_VIDEO_B_PATH, f"{HOME1}/pose/output/pose-estimation-synchronised-sample-b.json", f"{HOME1}/pose/output/right-shoulder-angle-b.xlsx", f"{HOME1}/pose/output/right-shoulder-angles-sample-b.mp4", f"{HOME1}/pose/output/right-shoulder-angles-sample-b.png", f"{HOME1}/pose/output")
    print("hello")
    #process_and_dump(SOURCE_VIDEO_C_PATH, f"{HOME1}/pose/output/pose-estimation-synchronised-sample-c.json", f"{HOME1}/pose/output/right-shoulder-angle-c.xlsx", f"{HOME1}/pose/output/right-shoulder-angles-sample-c.mp4", f"{HOME1}/pose/output/right-shoulder-angles-sample-c.png", f"{HOME1}/pose/output")
    print("hello")

    video_files = [
        "right-shoulder-angles-sample-b.mp4",
        "right-shoulder-angles-sample-b.mp4",
        "right-shoulder-angles-sample-c.mp4"
    ]

    for video_file in video_files:
        original_path = OUTPUT_DIRECTROY / video_file
        converted_path = OUTPUT_DIRECTROY / video_file.replace(".mp4", "2.mp4")

        # Check if the file exists before attempting conversion
        if original_path.exists():
            print(f"Re-encoding {original_path.name}")
            subprocess.run([
                FFMPEG_PATH,
                "-y", "-i", str(original_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                str(converted_path)
            ], check=True)

            # Remove the original file
           # os.remove(original_path)
        else:
            print(f"File not found: {original_path}")

    return "success"
@app.post("/anime/")
async def anime():
    print("HOME1 in ani",HOME1)
    #generate_animation()

    # Re-encode using FFmpeg to H.264 with faststart
    subprocess.run([
    r"C:\Users\Jagath\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe", "-y", "-i", 
    r"C:\Users\Jagath\Desktop\VisionPose\VisionPose\backend\pose\output\output_video.mp4",
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-movflags", "+faststart", 
    r"C:\Users\Jagath\Desktop\VisionPose\VisionPose\backend\pose\output\output_video2.mp4"
])

    return "success anime"

# @app.post("/animation/")
# async def animation():
#     print("HOME1 in ani",HOME1)
#     #generate_animation()

#     # Re-encode using FFmpeg to H.264 with faststart
#     # subprocess.run([
#     #     r"C:\Users\Jagath\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe", "-y", "-i", "C:\Users\Jagath\Desktop\VisionPose\VisionPose\backend\pose\output\output_video.mp4",
#     #     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
#     #     "-movflags", "+faststart", "C:\Users\Jagath\Desktop\VisionPose\VisionPose\backend\pose\output\output_video2.mp4"
#     # ])
#     return "success anime"
