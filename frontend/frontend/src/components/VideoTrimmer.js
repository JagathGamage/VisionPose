import { useState, useRef, useEffect } from "react";
import { Typography, Button, Grid, Slider, Box, Paper } from "@mui/material";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
import axios from "axios";
import { useLocation } from "react-router-dom";
const synced_1 = "/videos/synced_1.mp4";
const synced_2 = "/videos/synced_2.mp4";
const synced_3 = "/videos/synced_3.mp4";



const ffmpeg = new FFmpeg();
const uploadedVideos = [synced_1,synced_2,synced_3];

export default function VideoTrimmer() {
  const location = useLocation();
  const { selectedRequirement } = location.state || { selectedRequirement: "No requirement selected" };

  const [trimRanges, setTrimRanges] = useState([
    [0, 100],
    [0, 100],
    [0, 100],
  ]);
  const [frames, setFrames] = useState([[], [], []]);
  const [trimmedVideos, setTrimmedVideos] = useState([null, null, null]);

  const videoRefs = [useRef(null), useRef(null), useRef(null)];
  const [uploadStatus, setUploadStatus] = useState([false, false, false]); // Track uploads
  const [allUploaded, setAllUploaded] = useState(false);

  useEffect(() => {
    const loadFFmpeg = async () => {
      if (!ffmpeg.loaded) {
        await ffmpeg.load();
      }
    };
    loadFFmpeg();
  }, []);


  useEffect(() => {
    if (uploadStatus.every(status => status)) {
      setAllUploaded(true);
    }
  }, [uploadStatus]);
  
  // Extract frames from the video
  const extractFrames = (video, index) => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const extractedFrames = [];
    const duration = video.duration;
    const frameInterval = duration / 20; // Extract 20 frames

    let captureFrame = (time) => {
      video.currentTime = time;
      setTimeout(() => {
        canvas.width = video.videoWidth / 2; // Bigger frames
        canvas.height = video.videoHeight / 2;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        extractedFrames.push(canvas.toDataURL("image/png"));
        if (extractedFrames.length < 20) {
          captureFrame(time + frameInterval);
        } else {
          setFrames((prevFrames) => {
            const newFrames = [...prevFrames];
            newFrames[index] = extractedFrames;
            return newFrames;
          });
        }
      }, 200);
    };

    captureFrame(0);
  };

  // Handle Trim Range Change
  const handleTrimChange = (index, newValue) => {
    const newTrimRanges = [...trimRanges];
    newTrimRanges[index] = newValue;
    setTrimRanges(newTrimRanges);

    // Adjust video to the new start time
    const video = videoRefs[index].current;
    if (video) {
      const duration = video.duration;
      const newStart = (newValue[0] / 100) * duration;
      video.currentTime = newStart;
    }
  };

  // Handle video time updates to restrict play within the trim range
  const handleTimeUpdate = (index) => {
    const video = videoRefs[index].current;
    if (!video) return;

    const duration = video.duration;
    const [startPercent, endPercent] = trimRanges[index];
    const startTime = (startPercent / 100) * duration;
    const endTime = (endPercent / 100) * duration;

    if (video.currentTime >= endTime) {
      video.currentTime = startTime; // Restart at start of trimmed section
      video.play();
    }
  };

  // Trim the video using FFmpeg.wasm
  const handleTrim = async (index) => {
    if (!ffmpeg.loaded) await ffmpeg.load();

    const inputName = `input${index}.mp4`;
    const outputName = `trimmed${index}.mp4`;

    // Convert video to a format FFmpeg can process
    const videoBlob = await fetchFile(uploadedVideos[index]);
    await ffmpeg.writeFile(inputName, videoBlob);

    const video = videoRefs[index].current;
    const duration = video.duration;
    const startTime = (trimRanges[index][0] / 100) * duration;
    const endTime = (trimRanges[index][1] / 100) * duration;

    await ffmpeg.exec([
      "-i", inputName,
      "-ss", `${startTime}`,
      "-to", `${endTime}`,
      "-c", "copy",
      outputName
    ]);

    const data = await ffmpeg.readFile(outputName);
    const trimmedBlob = new Blob([data], { type: "video/mp4" });
    const url = URL.createObjectURL(new Blob([data], { type: "video/mp4" }));

    setTrimmedVideos((prev) => {
    
      const newVideos = [...prev];
      newVideos[index] = url;
      return newVideos;
    });

    // Send trimmed video to backend
    const formData = new FormData();
    formData.append("file", trimmedBlob, `trimmed${index}.mp4`);
    
    try {
      await axios.post("http://localhost:8000/uploadTrimedVideos/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(`Trimmed video ${index + 1} uploaded successfully`);

      setUploadStatus((prev) => {
        const newStatus = [...prev];
        newStatus[index] = true;
        return newStatus;
      });
            
    } catch (error) {
      console.error(`Error uploading video ${index + 1}:`, error);
    }
  };

  // Function to call processAndDump API after successful uploads
const callProcessAndDump = async () => {
  try {
      const response = await axios.post("http://localhost:8000/processAndDump");
      console.log("processAndDump triggered successfully:", response.data);
      alert("Processing and Dumping Completed!");
  } catch (error) {
      console.error("Error calling processAndDump:", error);
  }
};

  return (
    <Box sx={{ maxWidth: 1200, mx: "auto", mt: 4, p: 3 }}>
      <Typography variant="h5" fontWeight="bold" align="center" gutterBottom>
        Trim Videos & Highlight Frames
      </Typography>
      <Typography variant="h6" align="center">Requirement: {selectedRequirement}</Typography>
      <Typography variant="body2" color="textSecondary" align="center" mb={3}>
        Select the relevant part of each video using the sliders. The selected section is highlighted in the frame preview.
      </Typography>

      <Grid container spacing={4}>
        {uploadedVideos.map((video, index) => (
          <Grid item xs={12} sm={4} key={index}>
            <Paper elevation={3} sx={{ p: 3, textAlign: "center" }}>
              <Typography variant="h6">Video {index + 1}</Typography>
              <video
                ref={videoRefs[index]}
                controls
                src={video}
                width="100%"
                height="300px"
                style={{
                  marginTop: "10px",
                  borderRadius: "8px",
                  border: "2px solid #ccc",
                }}
                onLoadedMetadata={() => extractFrames(videoRefs[index].current, index)}
                onTimeUpdate={() => handleTimeUpdate(index)}
              />

              {/* Trim Slider */}
              <Slider
                value={trimRanges[index]}
                onChange={(e, newValue) => handleTrimChange(index, newValue)}
                valueLabelDisplay="auto"
                min={0}
                max={100}
                step={1}
                sx={{ mt: 2 }}
              />

              <Typography variant="body2" color="textSecondary">
                Trim from {trimRanges[index][0]}% to {trimRanges[index][1]}%
              </Typography>

              {/* Frame Preview */}
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

              {/* Trim Button */}
              <Button
                variant="contained"
                color="primary"
                sx={{ mt: 2 }}
                onClick={() => handleTrim(index)}
              >
                Trim Video
              </Button>

              {/* Trimmed Video Preview */}
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
      {/* Show "Process & Dump" button when all videos are uploaded */}
      {allUploaded && (
        <Box textAlign="center" mt={4}>
          <Button
            variant="contained"
            color="success"
            onClick={callProcessAndDump}
          >
            Process & Dump
          </Button>
        </Box>
      )}
    </Box>
  );
}
