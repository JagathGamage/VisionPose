// src/components/Sync.jsx
import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { Box, Typography, Button, Grid, Paper } from "@mui/material";
import axios from "axios";

export default function Sync() {
  const navigate = useNavigate();
  const location = useLocation();
  const { videos = [], projectName } = location.state || {};

  // We only handle exactly three videos here—but add guards if needed
  const videoRefs = [useRef(), useRef(), useRef()];
  const [frames, setFrames] = useState([[], [], []]);
  const [selectedFrames, setSelectedFrames] = useState([null, null, null]);

  useEffect(() => {
    // As soon as “videos” arrives (from the previous page), extract all frames for each video:
    videos.forEach((file, idx) => extractFrames(file, idx));
  }, [videos]);

  const extractFrames = (file, index) => {
    const video = document.createElement("video");
    video.src = URL.createObjectURL(file);
    video.crossOrigin = "anonymous";
    video.preload = "metadata";

    video.addEventListener("loadedmetadata", () => {
      const duration = video.duration;
      const frameRate = 25; // assume 25 FPS
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

  const handleSyncVideos = async () => {
    // 1) Make sure user clicked exactly one frame for each of the three videos:
    if (selectedFrames.some((f) => f === null)) {
      alert("Please select one frame from each of the three videos before syncing.");
      return;
    }

    const formData = new FormData();

    // 2) Append each file under the SAME key "file_paths", and pass its filename explicitly:
    videos.forEach((fileObj) => {
      formData.append("file_paths", fileObj, fileObj.name);
    });

    // 3) Append each sync time (as a string) under "sync_times"
    selectedFrames.forEach((frame) => {
      formData.append("sync_times", frame.time.toString());
    });

    try {
      const res = await axios.post("http://127.0.0.1:8000/sync/", formData, {
        // DO NOT set Content-Type manually! Let Axios set the proper boundary:
        timeout: 100000,
      });
      console.log("FastAPI responded with:", res.data);
      alert("Videos synced successfully!");
      navigate("/requirementSelector");
    } catch (err) {
      console.error("Sync error:", err);
      const detail = err.response?.data?.detail || err.message;
      alert("Sync failed: " + detail);
    }
  };

  return (
    <Box p={4}>
      <Typography variant="h5" mb={2}>
        Project: {projectName}
      </Typography>

      <Grid container spacing={4}>
        {videos.map((file, index) => (
          <Grid item xs={12} md={4} key={index}>
            <video
              ref={videoRefs[index]}
              src={URL.createObjectURL(file)}
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

      <Button
        variant="contained"
        color="primary"
        onClick={handleSyncVideos}
        sx={{ mt: 4 }}
      >
        Sync Videos
      </Button>
    </Box>
  );
}
