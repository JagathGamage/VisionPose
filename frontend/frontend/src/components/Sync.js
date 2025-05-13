// RequirementSelector.jsx
import { useLocation,useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { Box, Typography, Button, Grid, Paper } from "@mui/material";

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
      const interval = duration / 10; // 10 thumbnails per video
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const currentFrames = [];

      const capture = (time) => {
        video.currentTime = time;
      };

      video.addEventListener("seeked", function captureFrame() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL();
        currentFrames.push({ time: video.currentTime, image: dataUrl });

        if (currentFrames.length < 10) {
          capture(currentFrames.length * interval);
        } else {
          setFrames((prev) => {
            const updated = [...prev];
            updated[index] = currentFrames;
            return updated;
          });
          video.removeEventListener("seeked", captureFrame);
        }
      });

      capture(0);
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
      const res = await fetch("http://127.0.0.1:8000/sync/", {
        method: "POST",
        body: formData,
      });
  
      const data = await res.json();
      alert(data.message);
  
      // Redirect to Requirement Selector page with necessary state
      navigate("/requirementSelector", {
        state: { projectName, syncedVideos: videos },
      });
  
    } catch (err) {
      console.error(err);
      alert("Failed to sync videos.");
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
