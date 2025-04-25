import React, { useState, useRef } from "react";
import { Typography, Button, Grid, Box, Paper } from "@mui/material";
import { FFmpeg } from "@ffmpeg/ffmpeg";

import v1 from "../initialVideos/v1.mp4";
import v2 from "../initialVideos/v2.mp4";
import v3 from "../initialVideos/v3.mp4";

const ffmpeg = new FFmpeg({ log: true });
const uploadedVideos = [v1, v2, v3];

function TrimVideos() {
  const videoRefs = useRef(uploadedVideos.map(() => React.createRef()));
  const [trimRanges, setTrimRanges] = useState(uploadedVideos.map(() => [10, 90]));
  const [frames, setFrames] = useState(uploadedVideos.map(() => Array(20).fill("")));
  const [trimmedVideos, setTrimmedVideos] = useState(uploadedVideos.map(() => null));
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(null);

  const extractFrames = async (videoEl, index) => {
    const duration = videoEl.duration;
    const frameTimes = [];
    const totalFrames = 20;
    const step = duration / totalFrames;

    for (let i = 0; i < totalFrames; i++) {
      frameTimes.push(i * step);
    }

    // Wait for FFmpeg to load
    await ffmpeg.load();

    // Generate frame images using FFmpeg at specified times
    const frameUrls = await Promise.all(
      frameTimes.map(async (time) => {
        const imageBlob = await getFrameFromVideo(videoEl, time, index);
        return URL.createObjectURL(imageBlob);
      })
    );

    setFrames((prev) => {
      const updated = [...prev];
      updated[index] = frameUrls;
      return updated;
    });
  };

  const getFrameFromVideo = async (videoEl, time, index) => {
    const videoFile = await fetch(uploadedVideos[index]).then((res) => res.blob());
    const inputName = `input${index}.mp4`;
    const outputName = `frame${index}-${time}.png`;

    // Write the video file to FFmpeg's filesystem
    ffmpeg.FS("writeFile", inputName, await ffmpeg.fetchFile(videoFile));

    // Run FFmpeg to extract the frame at the specific time
    await ffmpeg.run(
      "-i", inputName,
      "-ss", `${time}`,  // Time for the frame extraction
      "-vframes", "1",   // Extract one frame
      "-vf", "scale=320:-1", // Resize for thumbnail
      outputName
    );

    // Read the frame data from FFmpeg's filesystem
    const data = ffmpeg.FS("readFile", outputName);
    return new Blob([data.buffer], { type: "image/png" });
  };

  const handleTrim = async (index) => {
    const videoFile = await fetch(uploadedVideos[index]).then((res) => res.blob());
    const inputName = `input${index}.mp4`;
    const outputName = `output${index}.mp4`;

    // Wait for FFmpeg to load
    await ffmpeg.load();

    // Write the video file to FFmpeg's filesystem
    ffmpeg.FS("writeFile", inputName, await ffmpeg.fetchFile(videoFile));

    const duration = videoRefs.current[index].current.duration;
    const [startPercent, endPercent] = trimRanges[index];
    const startTime = (startPercent / 100) * duration;
    const endTime = (endPercent / 100) * duration;

    // Run FFmpeg to trim the video
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

  const handleSync = () => {
    if (selectedFrameIndex === null) return;
    const totalFrames = 20;

    uploadedVideos.forEach((_, index) => {
      const videoEl = videoRefs.current[index].current;
      if (!videoEl) return;
      const duration = videoEl.duration;
      const time = (selectedFrameIndex / totalFrames) * duration;

      videoEl.currentTime = time;
      setTimeout(() => videoEl.play(), 200);
    });
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: "auto", mt: 4, p: 3 }}>
      <Typography variant="h5" fontWeight="bold" align="center" gutterBottom>
        Sync Videos
      </Typography>
      <Typography variant="body2" color="textSecondary" align="center" mb={3}>
        Select a frame to sync all videos at that moment.
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
              />

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
                  const isSelected = i === selectedFrameIndex;
                  return (
                    <img
                      key={i}
                      src={frame}
                      alt={`Frame ${i}`}
                      onClick={() => setSelectedFrameIndex(i)}
                      style={{
                        width: "100px",
                        height: "auto",
                        marginRight: "8px",
                        border: isSelected ? "3px solid green" : "2px solid gray",
                        borderRadius: "6px",
                        cursor: "pointer",
                      }}
                    />
                  );
                })}
              </Box>

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
          onClick={handleSync}
          disabled={selectedFrameIndex === null}
        >
          Sync
        </Button>
      </Box>
    </Box>
  );
}

export default TrimVideos;
