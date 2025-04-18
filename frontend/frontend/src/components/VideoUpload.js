import { useState } from "react";
import axios from "axios";
import { Box, Button, Card, CardContent, CardHeader, LinearProgress, Typography, TextField } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";


export default function VideoUpload() {
  const navigate = useNavigate();  // Initialize navigate
  const [videos, setVideos] = useState([]);
  const [projectName, setProjectName] = useState("");
  const [message, setMessage] = useState("");
  const [syncedFiles, setSyncedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length === 3) {
      setVideos(acceptedFiles);
    } else {
      alert("Please select exactly 3 video files.");
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    accept: "video/*",
    onDrop,
    multiple: true,
  });

  const handleUpload = async () => {
    if (videos.length !== 3 || !projectName) {
      alert("Enter a project name and select exactly 3 videos.");
      return;
    }

    setLoading(true);
    setMessage("");
    setProgress(0);

    const formData = new FormData();
    formData.append("project_name", projectName);
    videos.forEach((video) => formData.append("files", video));

    try {
      const res = await axios.post("http://127.0.0.1:8000/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percent);
        },
      });

      setMessage(res.data.message);
      setSyncedFiles(res.data.synced_files);

      // Redirect to requirement selection page after successful upload
      setTimeout(() => {
        navigate("/requirementSelector", { state: { projectName, syncedFiles: res.data.synced_files } });
      }, 2000); // Delay for 2 seconds to show success message

    } catch (error) {
      console.error("Upload failed:", error);
      setMessage("Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh" bgcolor="#f4f6f8">
      <Card sx={{ width: 450, p: 3, boxShadow: 3 }}>
        <CardHeader title="Upload & Sync Videos 🎥" sx={{ textAlign: "center" }} />
        <CardContent>
          <TextField
            fullWidth
            label="Project Name"
            variant="outlined"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            sx={{ mb: 2 }}
          />

          <Box
            {...getRootProps()}
            sx={{
              border: "2px dashed #aaa",
              borderRadius: 2,
              p: 4,
              textAlign: "center",
              cursor: "pointer",
              bgcolor: "#fafafa",
              "&:hover": { bgcolor: "#f0f0f0" },
            }}
          >
            <input {...getInputProps()} />
            <CloudUploadIcon fontSize="large" color="action" />
            <Typography variant="body2" color="textSecondary">
              Drag & drop 3 video files here or click to select
            </Typography>
          </Box>

          {videos.length > 0 && (
            <Box mt={2}>
              <Typography variant="subtitle2">Selected Videos:</Typography>
              {videos.map((file, index) => (
                <Typography key={index} variant="body2" color="textSecondary">
                  {file.name}
                </Typography>
              ))}
            </Box>
          )}

          {loading && (
            <Box mt={2}>
              <LinearProgress variant="determinate" value={progress} />
              <Typography variant="body2" color="textSecondary" textAlign="center">
                {progress}%
              </Typography>
            </Box>
          )}

          <Button
            fullWidth
            variant="contained"
            color="primary"
            startIcon={<CloudUploadIcon />}
            sx={{ mt: 3 }}
            onClick={handleUpload}
            disabled={loading}
          >
            {loading ? "Uploading..." : "Upload Videos"}
          </Button>

          {message && (
            <Typography variant="body2" color="green" textAlign="center" mt={2}>
              {message}
            </Typography>
          )}

          {syncedFiles.length > 0 && (
            <Box mt={3}>
              <Typography variant="subtitle2">Synced Videos:</Typography>
              {syncedFiles.map((file, index) => (
                <Typography key={index} variant="body2" color="textSecondary">
                  {file}
                </Typography>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
