import React, { useState, useRef, useEffect } from "react";
import { Typography, Button, Grid, Slider, Box, Paper } from "@mui/material";
import { FFmpeg } from "@ffmpeg/ffmpeg";

import v1 from "../videos/v1.mp4";
import v2 from "../videos/v2.mp4";
import v3 from "../videos/v3.mp4";

const ffmpeg = new FFmpeg({ log: true });
const uploadedVideos = [v1, v2, v3];

function TrimVideos() {
  const videoRefs = useRef(uploadedVideos.map(() => React.createRef()));
  const [trimRanges, setTrimRanges] = useState(uploadedVideos.map(() => [10, 90]));
  const [frames, setFrames] = useState(uploadedVideos.map(() => Array(20).fill(""))); // Mock frames
  const [trimmedVideos, setTrimmedVideos] = useState(uploadedVideos.map(() => null));

  // Dummy extractor (replace with real implementation)
  const extractFrames = (videoEl, index) => {
    // For now, mock the frames using thumbnails (not actual video parsing)
    const dummyFrames = Array(20).fill("https://via.placeholder.com/100");
    setFrames((prev) => {
      const updated = [...prev];
      updated[index] = dummyFrames;
      return updated;
    });
  };

  const handleTrimChange = (index, newValue) => {
    setTrimRanges((prev) => {
      const updated = [...prev];
      updated[index] = newValue;
      return updated;
    });
  };

  const handleTimeUpdate = (index) => {
    // Optional: live update based on current time
  };

  const handleTrim = async (index) => {
    const videoFile = await fetch(uploadedVideos[index]).then((res) => res.blob());
    const inputName = `input${index}.mp4`;
    const outputName = `output${index}.mp4`;

    await ffmpeg.load();

    ffmpeg.FS("writeFile", inputName, await ffmpeg.fetchFile(videoFile));

    const duration = videoRefs.current[index].current.duration;
    const [startPercent, endPercent] = trimRanges[index];
    const startTime = (startPercent / 100) * duration;
    const endTime = (endPercent / 100) * duration;

    await ffmpeg.run(
      "-i", inputName,
      "-ss", `${startTime}`,
      "-to", `${endTime}`,
      "-c:v", "copy",
      "-c:a", "copy",
      outputName
    );

    const data = ffmpeg.FS("readFile", outputName);
    const url = URL.createObjectURL(new Blob([data.buffer], { type: "video/mp4" }));

    setTrimmedVideos((prev) => {
      const updated = [...prev];
      updated[index] = url;
      return updated;
    });
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: "auto", mt: 4, p: 3 }}>
      <Typography variant="h5" fontWeight="bold" align="center" gutterBottom>
        Trim Videos 
      </Typography>
      <Typography variant="body2" color="textSecondary" align="center" mb={3}>
        Select the relevant part of each video using the sliders.
      </Typography>

      <Grid container spacing={4}>
        {uploadedVideos.map((video, index) => (
          <Grid item xs={12} sm={4} key={index}>
            <Paper elevation={3} sx={{ p: 3, textAlign: "center" }}>
              <Typography variant="h6">Video {index + 1}</Typography>
              <video
                ref={videoRefs.current[index]}
                controls
                src={video}
                width="100%"
                height="300px"
                style={{
                  marginTop: "10px",
                  borderRadius: "8px",
                  border: "2px solid #ccc",
                }}
                onLoadedMetadata={() => extractFrames(videoRefs.current[index].current, index)}
                onTimeUpdate={() => handleTimeUpdate(index)}
              />

              <Slider
                value={trimRanges[index]}
                onChange={(e, val) => handleTrimChange(index, val)}
                valueLabelDisplay="auto"
                min={0}
                max={100}
                step={1}
                sx={{ mt: 2 }}
              />
              <Typography variant="body2" color="textSecondary">
                Trim from {trimRanges[index][0]}% to {trimRanges[index][1]}%
              </Typography>

              <Box
                sx={{
                  display: "flex",
                  overflowX: "auto",
                  mt: 2,
                  p: 2,
                  bgcolor: "#f5f5f5",
                  borderRadius: "8px",
                  border: "2px solid #ddd",
                }}
              >
                {frames[index].map((frame, i) => {
                  const isHighlighted =
                    i >= Math.floor((trimRanges[index][0] / 100) * 20) &&
                    i <= Math.ceil((trimRanges[index][1] / 100) * 20);
                  return (
                    <img
                      key={i}
                      src={frame}
                      alt={`Frame ${i}`}
                      style={{
                        width: "100px",
                        height: "auto",
                        marginRight: "8px",
                        border: isHighlighted ? "3px solid red" : "2px solid gray",
                        borderRadius: "6px",
                      }}
                    />
                  );
                })}
              </Box>

              <Button
                variant="contained"
                color="primary"
                sx={{ mt: 2 }}
                onClick={() => handleTrim(index)}
              >
                Trim Video
              </Button>

              {trimmedVideos[index] && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1">Trimmed Video</Typography>
                  <video src={trimmedVideos[index]} controls width="100%" />
                  <a href={trimmedVideos[index]} download={`trimmed${index + 1}.mp4`}>
                    <Button variant="outlined" color="secondary" sx={{ mt: 1 }}>
                      Download Trimmed Video
                    </Button>
                  </a>
                </Box>
              )}
            </Paper>
          </Grid>
        ))}
      </Grid>
      <Box display="flex" justifyContent="center" mt={2}>
      <Button
                variant="contained"
                color="primary"
                sx={{ mt: 2 }}
                
              >
                Sync
              </Button>
              </Box>
    </Box>
  );
}

export default TrimVideos;



