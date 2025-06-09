import { useState, useEffect } from "react";
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Container,
  IconButton,
  Tooltip,
  Snackbar,
  Alert,
  Fade,
  CircularProgress,
  Divider
} from "@mui/material";
import { Link } from "react-router-dom";
import DownloadIcon from "@mui/icons-material/Download";
import ShareIcon from "@mui/icons-material/Share";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import VolumeOffIcon from "@mui/icons-material/VolumeOff";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import DashboardIcon from "@mui/icons-material/Dashboard";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";

export default function FinalVideoPage() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarSeverity, setSnackbarSeverity] = useState("success");
  
  // Simulate video loading
  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  // Handle video play/pause
  const togglePlay = () => {
    const videoElement = document.getElementById("final-video");
    if (videoElement) {
      if (isPlaying) {
        videoElement.pause();
      } else {
        videoElement.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Handle mute/unmute
  const toggleMute = () => {
    const videoElement = document.getElementById("final-video");
    if (videoElement) {
      videoElement.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  // Handle fullscreen
  const toggleFullscreen = () => {
    const videoElement = document.getElementById("final-video");
    if (videoElement) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        videoElement.requestFullscreen();
      }
    }
  };

  // Handle download
  const handleDownload = () => {
    // In a real app, this would trigger the actual download
    setSnackbarMessage("Download started");
    setSnackbarSeverity("success");
    setShowSnackbar(true);
  };

  // Handle share
  const handleShare = () => {
    // In a real app, this would open a share dialog
    setSnackbarMessage("Share link copied to clipboard");
    setSnackbarSeverity("success");
    setShowSnackbar(true);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4, display: "flex", flexDirection: "column", alignItems: "center" }}>
        <Typography 
          variant="h3" 
          component="h1" 
          fontWeight="700" 
          align="center"
          sx={{ 
            mb: 1,
            background: "linear-gradient(90deg, #3a7bd5, #00d2ff)",
            backgroundClip: "text",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            textShadow: "0px 0px 1px rgba(0,0,0,0.05)"
          }}
        >
          Your Video is Ready
        </Typography>
        <Typography 
          variant="h6" 
          color="text.secondary" 
          align="center"
          sx={{ maxWidth: "700px", mb: 2 }}
        >
        Processed Final Video 
        </Typography>
        <Box 
          sx={{ 
            display: "flex", 
            alignItems: "center", 
            bgcolor: "success.light", 
            color: "success.contrastText",
            borderRadius: 2,
            px: 2,
            py: 0.5
          }}
        >
          <CheckCircleOutlineIcon fontSize="small" sx={{ mr: 1 }} />
          <Typography variant="body2">Successfully rendered in high quality</Typography>
        </Box>
      </Box>

      {/* Video Player Section */}
      <Paper 
        elevation={6} 
        sx={{ 
          borderRadius: 3, 
          overflow: "hidden",
          mb: 4,
          background: "#1a1a2e",
          position: "relative"
        }}
      >
        {loading ? (
          <Box 
            sx={{ 
              display: "flex", 
              justifyContent: "center", 
              alignItems: "center",
              minHeight: "400px",
              bgcolor: "rgba(0,0,0,0.03)"
            }}
          >
            <CircularProgress />
            <Typography sx={{ ml: 2 }}>Loading your video...</Typography>
          </Box>
        ) : (
          <Fade in={!loading}>
            <Box>
              {/* Video Element */}
              <Box sx={{ position: "relative" }}>
                <video
                  id="final-video"
                  width="75%"
                  src="http://127.0.0.1:8000/animation/output_video.mp4"
                  poster="/path/to/thumbnail.jpg"
                  preload="auto"
                  style={{ display: "block" }}
                  onEnded={() => setIsPlaying(false)}
                />

                {/* Custom Video Controls */}
                <Box 
                  sx={{ 
                    position: "absolute", 
                    bottom: 0, 
                    left: 0, 
                    right: 0,
                    background: "linear-gradient(to top, rgba(0,0,0,0.7), transparent)",
                    p: 2,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between"
                  }}
                >
                  <Box sx={{ display: "flex", alignItems: "center" }}>
                    <IconButton 
                      onClick={togglePlay} 
                      sx={{ color: "white" }}
                    >
                      {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                    </IconButton>
                    <IconButton 
                      onClick={toggleMute} 
                      sx={{ color: "white" }}
                    >
                      {isMuted ? <VolumeOffIcon /> : <VolumeUpIcon />}
                    </IconButton>
                  </Box>
                  <Box>
                    <IconButton 
                      onClick={toggleFullscreen} 
                      sx={{ color: "white" }}
                    >
                      <FullscreenIcon />
                    </IconButton>
                  </Box>
                </Box>
              </Box>

              {/* Video Actions Bar */}
              <Box 
                sx={{ 
                  display: "flex", 
                  justifyContent: "space-between",
                  alignItems: "center",
                  bgcolor: "#f8f9fa", 
                  p: 2
                }}
              >
                <Typography variant="subtitle1" fontWeight="medium">
                  Final_Video_v1.mp4
                </Typography>
                <Box>
                  <Tooltip title="Download video">
                    <IconButton 
                      color="primary" 
                      onClick={handleDownload}
                      sx={{ mr: 1 }}
                    >
                      <DownloadIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Share video">
                    <IconButton 
                      color="primary"
                      onClick={handleShare}
                    >
                      <ShareIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
            </Box>
          </Fade>
        )}
      </Paper>

      {/* Video Details */}
      <Paper 
        elevation={2} 
        sx={{ 
          borderRadius: 2, 
          overflow: "hidden",
          mb: 4,
          bgcolor: "#fbfbfb"
        }}
      >
        <Box sx={{ p: 3 }}>
          <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
            Video Details
          </Typography>
          
          <Box sx={{ display: "flex", flexDirection: { xs: "column", md: "row" }, gap: 4 }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Resolution
              </Typography>
              <Typography variant="body1" fontWeight="medium">
                1920 x 1080 (Full HD)
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Duration
              </Typography>
              <Typography variant="body1" fontWeight="medium">
                2:34 minutes
              </Typography>
            </Box>
            
            <Box sx={{ flex: 1 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                File Format
              </Typography>
              <Typography variant="body1" fontWeight="medium">
                MP4 (H.264)
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                File Size
              </Typography>
              <Typography variant="body1" fontWeight="medium">
                86.4 MB
              </Typography>
            </Box>
          </Box>
        </Box>
      </Paper>

      {/* Actions Footer */}
      <Box sx={{ display: "flex", justifyContent: "center", gap: 2, mt: 6 }}>
        <Button
          variant="outlined"
          size="large"
          component={Link}
          to="/graphs"
          startIcon={<DashboardIcon />}
          sx={{ 
            borderRadius: 2,
            px: 3,
            py: 1
          }}
        >
          Back to Dashboard
        </Button>
        
        <Button
          variant="contained"
          size="large"
          component={Link}
          to="/create-new"
          sx={{ 
            borderRadius: 2,
            px: 3,
            py: 1,
            background: "linear-gradient(90deg, #3a7bd5, #00d2ff)",
            boxShadow: "0 4px 20px rgba(0, 0, 0, 0.1)"
          }}
        >
          Create New Video
        </Button>
      </Box>

      {/* Notification */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={4000}
        onClose={() => setShowSnackbar(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert 
          onClose={() => setShowSnackbar(false)} 
          severity={snackbarSeverity}
          variant="filled"
          sx={{ width: "100%" }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Container>
  );
}