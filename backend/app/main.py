# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from uuid import uuid4

from pose.DPoseEstimationUsingYOLOv7 import process_and_dump ,generate_animation


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

# This folder is only used for temporarily storing uploads before we process them:
UPLOAD_DIR = os.path.join(HOME, "uploaded_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory to save projects and serve static files
PROJECTS_DIR = "projects"
STATIC_FILES_DIR = "C:/Users/ucsc/Desktop/vs/VisionPose/frontend/frontend/public/videos"  # Specify your frontend folder here

# Ensure directories exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(STATIC_FILES_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=STATIC_FILES_DIR), name="videos")


@app.get("/")
async def hello():
    return {"message": "hello from backend"}


@app.post("/sync/")
async def sync_videos(
    file_paths: List[UploadFile] = File(...),
    sync_times: List[float] = Form(...),
):
    """
    Expect exactly 3 files (as UploadFile) and exactly 3 sync_times (floats).
    We will cut each video so that all three start at the same relative timestamp.
    """
    # DEBUG: print what FastAPI actually received
    print("=== /sync/ called ===")
    print("Received file_paths:", [f.filename for f in file_paths])
    print("Received sync_times:", sync_times)

    if len(file_paths) != 3 or len(sync_times) != 3:
        raise HTTPException(
            status_code=400,
            detail="Exactly 3 videos and 3 sync times are required",
        )

    temp_video_paths: List[str] = []
    synced_paths: List[str] = []

    try:
        # 1) Save each incoming UploadFile into UPLOAD_DIR as a temp file
        for upload in file_paths:
            unique_name = f"temp_{uuid.uuid4().hex}_{upload.filename}"
            temp_path = os.path.join(UPLOAD_DIR, unique_name)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            temp_video_paths.append(temp_path)

        # 2) Find the smallest sync time, so we know how much to trim from each
        min_sync_time = min(sync_times)

        # 3) For each video, open it with OpenCV, seek to (sync_time - min_sync_time)*fps, and write the remainder
        for idx, (path, sync_time) in enumerate(zip(temp_video_paths, sync_times)):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise HTTPException(
                    status_code=400, detail=f"Could not open video file {path}"
                )

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            output_filename = f"synced_{idx+1}.mp4"

            # Write to BOTH the SYNCED_DIR (for React) AND STATIC_FILES_DIR (same path, but just in case)
            for output_dir in (SYNCED_DIR, STATIC_FILES_DIR):
                output_path = os.path.join(output_dir, output_filename)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not create output video {output_path}",
                    )

                # Compute the start frame based on (sync_time - min_sync_time)
                start_frame = int((sync_time - min_sync_time) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                # Copy remaining frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()

            synced_paths.append(f"/videos/{output_filename}")
            cap.release()

        return JSONResponse(
            {
                "message": "Videos synced successfully",
                "output_paths": synced_paths,
            }
        )

    finally:
        # Clean up any temp files we created
        for temp_path in temp_video_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Warning: could not delete temp file {temp_path}: {e}")


# ----------------------------
# (Existing trim‐upload endpoints follow—unchanged)
# ----------------------------
TRIM_DIR = os.path.join(HOME, "pose", "input")
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
