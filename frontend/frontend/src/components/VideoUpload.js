import { useState } from "react";
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  CircularProgress,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";

export default function VideoUpload() {
  const navigate = useNavigate();
  const [videos, setVideos] = useState([]);
  const [projectName, setProjectName] = useState("");
  const [loading, setLoading] = useState(false);  // <-- added loading state

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

  const handleProceed = async () => {
    if (videos.length !== 3 || !projectName) {
      alert("Enter a project name and select exactly 3 videos.");
      return;
    }

    setLoading(true);  // <-- start loading spinner

    const formData = new FormData();
    formData.append("project_name", projectName);
    videos.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch("http://127.0.0.1:8000/upload/", {
        method: "POST",
        body: formData,
      });

      // Assume upload success if response.ok
      if (response.ok) {
        console.log("Uploaded successfully:");
        navigate("/sync", {
          state: { projectName, videos },
        });
      } else {
        alert("Upload failed.");
      }
    } catch (err) {
      console.error("Error uploading videos:", err);
      alert("Error uploading videos.");
    } finally {
      setLoading(false);  // <-- stop loading spinner
    }
  };

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
      bgcolor="#f4f6f8"
    >
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
            disabled={loading} // disable input while loading
          />
          <Box
            {...getRootProps()}
            sx={{
              border: "2px dashed #aaa",
              borderRadius: 2,
              p: 4,
              textAlign: "center",
              cursor: loading ? "not-allowed" : "pointer", // show not-allowed cursor if loading
              bgcolor: "#fafafa",
              "&:hover": { bgcolor: loading ? "#fafafa" : "#f0f0f0" },
            }}
          >
            <input {...getInputProps()} disabled={loading} />
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

          <Button
            fullWidth
            variant="contained"
            color="primary"
            startIcon={loading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
            sx={{ mt: 3 }}
            onClick={handleProceed}
            disabled={loading} // disable button while loading
          >
            {loading ? "Uploading..." : "Proceed to Sync"}
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
}
