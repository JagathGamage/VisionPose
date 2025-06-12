// src/components/Sync.jsx
import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import {
  Box,
  Typography,
  Button,
  Grid,
  Paper,
  CircularProgress,
} from "@mui/material";
import axios from "axios";

export default function Sync() {
  const navigate = useNavigate();
  const location = useLocation();
  const { videos = [], projectName } = location.state || {};

  const videoPaths = ["http://127.0.0.1:8000/uploaded_videos/video_1_formatted.mp4", "http://127.0.0.1:8000/uploaded_videos/video_2_formatted.mp4", "http://127.0.0.1:8000/uploaded_videos/video_3_formatted.mp4"];
  const videoRefs = [useRef(), useRef(), useRef()];
  const [frames, setFrames] = useState([[], [], []]);
  const [selectedFrames, setSelectedFrames] = useState([null, null, null]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (videoPaths.length === 3) {
      let processed = 0;
      videoPaths.forEach((url, idx) =>
        extractFramesFromURL(url, idx, () => {
          processed += 1;
          if (processed === 3) {
            setLoading(false);
          }
        })
      );
    }
  }, []);

  const extractFramesFromURL = async (videoUrl, index, onComplete) => {
  const response = await fetch(videoUrl);
  const blob = await response.blob();
  extractFrames(blob, index, onComplete); // reuse existing method
  };


  const extractFrames = (file, index, onComplete) => {
    const video = document.createElement("video");
    video.src = URL.createObjectURL(file);
    video.crossOrigin = "anonymous";
    video.preload = "metadata";

    video.addEventListener("loadedmetadata", () => {
      const duration = video.duration;
      const frameRate = 10;
      const interval = 1 / frameRate;
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const collected = [];
      let currentTime = 0;

      const seekAndCapture = () => {
        if (currentTime > duration) {
          setFrames((prev) => {
            const copy = [...prev];
            copy[index] = collected;
            return copy;
          });
          onComplete(); // Notify when done
          return;
        }
        video.currentTime = currentTime;
      };

      const onSeeked = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL();
        collected.push({ time: video.currentTime, image: dataUrl });
        currentTime += interval;
        seekAndCapture();
      };

      video.addEventListener("seeked", onSeeked);
      seekAndCapture();
    });
  };

  const handleSelectFrame = (videoIndex, frame) => {
    const copy = [...selectedFrames];
    copy[videoIndex] = frame;
    setSelectedFrames(copy);
  };

  const handleSyncVideos = async (e) => {
  e?.preventDefault(); // prevent page reload

  if (selectedFrames.some((f) => f === null)) {
    alert("Please select one frame from each of the three videos before syncing.");
    return;
  }

  const formData = new FormData();
  videos.forEach((fileObj) => {
    formData.append("file_paths", fileObj, fileObj.name);
  });
  selectedFrames.forEach((frame) => {
    formData.append("sync_times", frame.time.toString());
  });

  try {
    const res = await axios.post("http://127.0.0.1:8000/sync/", formData, {
      timeout: 100000,
    });
    alert("Videos synced successfully!");
    navigate("/requirementSelector"); // should trigger route change
  } catch (err) {
    const detail = err.response?.data?.detail || err.message;
    alert("Sync failed: " + detail);
  }
};


  return (
    <Box p={4}>
      <Typography variant="h5" mb={2}>
        Project: {projectName}
      </Typography>

      {loading ? (
        <Box display="flex" justifyContent="center" mt={4}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <Grid container spacing={4}>
            {videoPaths.map((file, index) => (
              <Grid item xs={12} md={4} key={index}>
                <video
                  ref={videoRefs[index]}
                  src={file}
                  controls
                  width="100%"
                />
                <Typography variant="subtitle2" mt={1}>
                  Select Frame:
                </Typography>
                <Box
                  display="flex"
                  overflow="auto"
                  gap={1}
                  mt={1}
                  sx={{
                    flexDirection: "row",
                    overflowX: "auto",
                    whiteSpace: "nowrap",
                    pb: 1,
                  }}
                >
                  {frames[index].map((frame, idx) => (
                    <Paper
                      key={idx}
                      elevation={
                        selectedFrames[index]?.time === frame.time ? 4 : 1
                      }
                      onClick={() => handleSelectFrame(index, frame)}
                      sx={{
                        border:
                          selectedFrames[index]?.time === frame.time
                            ? "2px solid blue"
                            : "1px solid #ccc",
                        cursor: "pointer",
                        p: 0.5,
                      }}
                    >
                      <img
                        src={frame.image}
                        alt={`frame-${index}-${idx}`}
                        width="100"
                      />
                    </Paper>
                  ))}
                </Box>
              </Grid>
            ))}
          </Grid>

          <form onSubmit={handleSyncVideos}>
            <Button type="submit" variant="contained">Sync Videos</Button>
           
          </form>
          {/* <Button
            variant="outlined"
            color="secondary"
            onClick={() => navigate("/requirementSelector")}
            sx={{ mt: 2, ml: 2 }}
          >
            Go to Requirement Selector
          </Button> */}

        </>
      )}
    </Box>
  );
}
