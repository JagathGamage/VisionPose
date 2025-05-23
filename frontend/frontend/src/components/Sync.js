// RequirementSelector.jsx
import { useLocation,useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { Box, Typography, Button, Grid, Paper } from "@mui/material";
import axios from "axios";

export default function Sync() {
  const navigate = useNavigate();
  const location = useLocation();
  const { videos = [], projectName } = location.state || {};
  const videoRefs = [useRef(), useRef(), useRef()];
  const [frames, setFrames] = useState([[], [], []]);
  const [selectedFrames, setSelectedFrames] = useState([null, null, null]);

  useEffect(() => {
    videos.forEach((video, i) => extractFrames(video, i));
  }, [videos]);

  const extractFrames = (file, index) => {
    const video = document.createElement("video");
    video.src = URL.createObjectURL(file);
    video.crossOrigin = "anonymous";
    video.preload = "metadata";
  
    video.addEventListener("loadedmetadata", () => {
      const duration = video.duration;
      const frameRate = 25; // Assume 25 FPS; you can change this
      const interval = 1 / frameRate;
  
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const currentFrames = [];
  
      let currentTime = 0;
  
      const seekAndCapture = () => {
        if (currentTime > duration) {
          setFrames((prev) => {
            const updated = [...prev];
            updated[index] = currentFrames;
            return updated;
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
        currentFrames.push({ time: video.currentTime, image: dataUrl });
  
        currentTime += interval;
        seekAndCapture(); // Move to next frame
      };
  
      video.addEventListener("seeked", onSeeked);
      seekAndCapture();
    });
  };
  
  

  const handleSelectFrame = (videoIndex, frame) => {
    const updated = [...selectedFrames];
    updated[videoIndex] = frame;
    setSelectedFrames(updated);
  };

  const handleSyncVideos = async () => {
  if (selectedFrames.some((frame) => !frame)) {
    alert("Select a frame from each video before syncing.");
    return;
  }

  const formData = new FormData();

  videos.forEach((videoFile) => {
    formData.append("file_paths", videoFile);
  });

  selectedFrames.forEach((frame) => {
    formData.append("sync_times", frame.time);
  });

  try {
    const res = await axios.post("http://127.0.0.1:8000/sync/", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    timeout: 100000, // 5 minutes in milliseconds
  });

    console.log("Response:", res.data);
    alert("Success");
    navigate("/requirementSelector");
  } catch (err) {
    console.error("Sync error:", err);
    alert("Sync failed: " + err.message);
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
                  elevation={selectedFrames[index]?.time === frame.time ? 4 : 1}
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
                  <img src={frame.image} alt={`frame-${idx}`} width="100" />
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
