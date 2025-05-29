import { useState, useEffect } from "react";
import {
  Box,
  Grid,
  Card,
  CardActionArea,
  CardMedia,
  CardContent,
  Typography,
  Dialog,
  IconButton,
  Stack,
  Button,
  Divider,
  Container,
  Paper,
  Chip,
  LinearProgress,
  Fade,
  Tooltip,
  Zoom,
  CircularProgress
} from "@mui/material";
import { 
  ShowChart, 
  Close, 
  Fullscreen, 
  Download, 
  FileDownload, 
  PlayArrow, 
  Info,
  DateRange,
  BarChart,
  Timeline,
  Speed,
  CompareArrows
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

// Import graph images
import LEFT from "../graphs/LEFT.png";
import MIDDLE from "../graphs/MIDDLE.png";
import RIGHT from "../graphs/RIGHT.png";
import COMBINED from "../graphs/COMBINED.png";

const graphImages = [
  { 
    src: LEFT, 
    name: "LEFT.png", 
    desc: "Left trajectory over time",
    category: "Left Analysis",
    icon: <Timeline />,
    lastUpdated: "May 14, 2025"
  },
  { 
    src: MIDDLE, 
    name: "MIDDLE.png", 
    desc: "Velocity & acceleration metrics", 
    category: "Middle Analysis",
    icon: <Speed />,
    lastUpdated: "May 14, 2025"
  },
  { 
    src: RIGHT, 
    name: "RIGHT.png", 
    desc: "Comparative right-side patterns", 
    category: "Right Analysis",
    icon: <CompareArrows />,
    lastUpdated: "May 13, 2025"
  },
  { 
    src: COMBINED, 
    name: "COMBINED.png", 
    desc: "All patterns integrated", 
    category: "Integrated View",
    icon: <BarChart />,
    lastUpdated: "May 12, 2025"
  },
];

export default function GraphDashboard() {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [hoveredGraph, setHoveredGraph] = useState(null);
  const navigate = useNavigate();

  // Simulate progress for video generation
  useEffect(() => {
    if (isGenerating) {
      const timer = setInterval(() => {
        setGenerationProgress((prevProgress) => {
          const newProgress = prevProgress + 15;
          if (newProgress >= 100) {
            clearInterval(timer);
            setTimeout(() => {
              console.log("Video generated successfully!");
              navigate("/final-video");
            }, 500);
            return 100;
          }
          return newProgress;
        });
      }, 500);

      return () => {
        clearInterval(timer);
      };
    }
  }, [isGenerating, navigate]);

  const generateVideo = () => {
    setIsGenerating(true);
  };

  const handleOpen = (graph) => {
    setSelected(graph);
    setOpen(true);
  };

  const handleClose = () => setOpen(false);

  const handleDownload = (event, graph) => {
    event.stopPropagation();
    // In a real app, this would trigger an actual download
    console.log(`Downloading ${graph.name}`);
    
    // Create a download link
    const link = document.createElement('a');
    link.href = graph.src;
    link.download = graph.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Paper
        elevation={0}
        sx={{
          mb: 6,
          backgroundImage: "linear-gradient(135deg, #3f51b5 0%, #2196f3 100%)",
          borderRadius: 4,
          py: 4,
          position: "relative",
          overflow: "hidden"
        }}
      >
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            opacity: 0.1,
            backgroundImage: "url('data:image/svg+xml,%3Csvg width=\"100\" height=\"100\" viewBox=\"0 0 100 100\" xmlns=\"http://www.w3.org/2000/svg\"%3E%3Cpath d=\"M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z\" fill=\"%23ffffff\" fill-opacity=\"1\" fill-rule=\"evenodd\"/%3E%3C/svg%3E')",
          }}
        />
        <Container maxWidth="lg">
          <Stack
            direction="column"
            alignItems="center"
            spacing={2}
            sx={{ position: "relative", zIndex: 1 }}
          >
            <Typography 
              variant="h3" 
              fontWeight="800" 
              color="white"
              sx={{
                textShadow: "0px 2px 4px rgba(0,0,0,0.2)",
                letterSpacing: 1
              }}
            >
              Motion Analysis Dashboard
            </Typography>
            <Typography variant="h6" color="white" align="center" sx={{ maxWidth: "700px", opacity: 0.9 }}>
              Comprehensive visualization of motion patterns and trajectories
            </Typography>
            <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
              <Chip 
                icon={<DateRange sx={{ color: "#fff !important" }} />} 
                label="Last updated: May 14, 2025" 
                sx={{ 
                  bgcolor: "rgba(255,255,255,0.2)", 
                  color: "white",
                  "& .MuiChip-label": { fontWeight: 500 }
                }} 
              />
              <Chip 
                icon={<Info sx={{ color: "#fff !important" }} />} 
                label="4 Visualizations Available" 
                sx={{ 
                  bgcolor: "rgba(255,255,255,0.2)", 
                  color: "white",
                  "& .MuiChip-label": { fontWeight: 500 }
                }} 
              />
            </Stack>
          </Stack>
        </Container>
      </Paper>

      {/* Subtitle with Explanation */}
      <Box sx={{ mb: 4, textAlign: "center" }}>
        <Typography variant="h5" fontWeight="medium" gutterBottom>
          Motion Analysis Visualizations
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 800, mx: "auto" }}>
          These visualizations provide detailed insights into movement patterns, velocity metrics, and 
          trajectory analysis. Select any visualization to examine in detail.
        </Typography>
      </Box>

      {/* Grid of Cards */}
      <Grid container spacing={3} sx={{ mb: 6 }}>
        {graphImages.map((graph, index) => (
          <Grid item xs={12} sm={6} md={3} key={graph.name}>
            <Card
              onMouseEnter={() => setHoveredGraph(index)}
              onMouseLeave={() => setHoveredGraph(null)}
              sx={{
                borderRadius: 3,
                overflow: "hidden",
                transition: "all 0.3s ease",
                transform: hoveredGraph === index ? "translateY(-8px)" : "none",
                boxShadow: hoveredGraph === index 
                  ? "0 12px 28px rgba(0, 0, 0, 0.15)" 
                  : "0 2px 8px rgba(0, 0, 0, 0.08)",
                height: "100%",
                display: "flex",
                flexDirection: "column"
              }}
            >
              <Box sx={{ position: "relative" }}>
                <CardActionArea onClick={() => handleOpen(graph)}>
                  <CardMedia
                    component="img"
                    image={graph.src}
                    alt={graph.name}
                    sx={{
                      height: 200,
                      objectFit: "cover"
                    }}
                  />
                  <Box
                    sx={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      background: "linear-gradient(to bottom, rgba(0,0,0,0) 70%, rgba(0,0,0,0.7) 100%)",
                    }}
                  />
                </CardActionArea>
                
                {/* Overlay actions */}
                <Fade in={hoveredGraph === index}>
                  <Box
                    sx={{
                      position: "absolute",
                      top: 8,
                      right: 8,
                      display: "flex",
                      gap: 1
                    }}
                  >
                    <Tooltip title="View full size" arrow>
                      <IconButton
                        size="small"
                        onClick={() => handleOpen(graph)}
                        sx={{ 
                          bgcolor: "rgba(255,255,255,0.9)",
                          "&:hover": { bgcolor: "rgba(255,255,255,1)" }
                        }}
                      >
                        <Fullscreen fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download graph" arrow>
                      <IconButton
                        size="small"
                        onClick={(e) => handleDownload(e, graph)}
                        sx={{ 
                          bgcolor: "rgba(255,255,255,0.9)",
                          "&:hover": { bgcolor: "rgba(255,255,255,1)" }
                        }}
                      >
                        <FileDownload fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Fade>
                
                {/* Category tag */}
                <Box
                  sx={{
                    position: "absolute",
                    bottom: 12,
                    left: 12,
                    bgcolor: "rgba(0,0,0,0.6)",
                    color: "white",
                    borderRadius: 2,
                    px: 1,
                    py: 0.5,
                    display: "flex",
                    alignItems: "center",
                    gap: 0.5
                  }}
                >
                  {graph.icon}
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    {graph.category}
                  </Typography>
                </Box>
              </Box>
              
              <CardContent sx={{ flexGrow: 1, display: "flex", flexDirection: "column" }}>
                <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
                  {graph.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {graph.desc}
                </Typography>
                <Box sx={{ mt: "auto", display: "flex", alignItems: "center", gap: 1 }}>
                  <DateRange fontSize="small" color="action" />
                  <Typography variant="caption" color="text.secondary">
                    {graph.lastUpdated}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Video Generation Section */}
      <Paper
        elevation={2}
        sx={{
          borderRadius: 3,
          p: 4,
          mt: 4,
          mb: 6,
          backgroundImage: "linear-gradient(to right, #f5f7fa 0%, #e4e7eb 100%)"
        }}
      >
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={7}>
            <Typography variant="h5" fontWeight="bold" gutterBottom>
              Ready to Generate Your Analysis Video?
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              Combine all visualizations into a comprehensive analysis video with detailed insights and smooth transitions.
            </Typography>
            <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
              <Button
                variant="contained"
                onClick={generateVideo}
                disabled={isGenerating}
                startIcon={isGenerating ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
                sx={{
                  px: 4,
                  py: 1.5,
                  fontSize: "16px",
                  fontWeight: "bold",
                  borderRadius: 2,
                  background: "linear-gradient(90deg, #4f46e5, #3b82f6)",
                  boxShadow: "0 10px 15px -3px rgba(59, 130, 246, 0.5)",
                  "&:hover": {
                    background: "linear-gradient(90deg, #6366f1, #60a5fa)",
                  },
                }}
              >
                {isGenerating ? "Generating..." : "Generate Video Analysis"}
              </Button>
              <Button
                variant="outlined"
                disabled={isGenerating}
                startIcon={<Download />}
                sx={{
                  px: 3,
                  py: 1.5,
                  fontSize: "16px",
                  fontWeight: "medium",
                  borderRadius: 2,
                  borderWidth: 2,
                  "&:hover": {
                    borderWidth: 2,
                  },
                }}
              >
                Export All Graphs
              </Button>
            </Box>
          </Grid>
          <Grid item xs={12} md={5}>
            {isGenerating ? (
              <Paper 
                elevation={0} 
                sx={{ 
                  bgcolor: "rgba(255, 255, 255, 0.7)", 
                  p: 3, 
                  borderRadius: 2 
                }}
              >
                <Stack spacing={2}>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="body1" fontWeight="medium">
                      Generation Progress
                    </Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {Math.round(generationProgress)}%
                    </Typography>
                  </Stack>
                  <LinearProgress 
                    variant="determinate" 
                    value={generationProgress} 
                    sx={{ 
                      height: 10, 
                      borderRadius: 5,
                      backgroundColor: "rgba(0, 0, 0, 0.08)",
                      "& .MuiLinearProgress-bar": {
                        backgroundColor: "#4f46e5",
                      }
                    }}
                  />
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                    {generationProgress < 25 && (
                      <Chip size="small" label="Processing graph data..." />
                    )}
                    {generationProgress >= 25 && generationProgress < 50 && (
                      <Chip size="small" label="Building visual sequences..." />
                    )}
                    {generationProgress >= 50 && generationProgress < 75 && (
                      <Chip size="small" label="Generating insights..." />
                    )}
                    {generationProgress >= 75 && generationProgress < 100 && (
                      <Chip size="small" label="Finalizing video render..." />
                    )}
                    {generationProgress >= 100 && (
                      <Chip 
                        size="small" 
                        color="success" 
                        label="Video generation complete!" 
                      />
                    )}
                  </Box>
                </Stack>
              </Paper>
            ) : (
              <Box 
                sx={{ 
                  display: "flex", 
                  justifyContent: "center", 
                  alignItems: "center",
                  p: 2
                }}
              >
                <img 
                  src="/api/placeholder/400/200" 
                  alt="Video preview" 
                  style={{ 
                    maxWidth: "100%", 
                    borderRadius: "8px",
                    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)"
                  }} 
                />
              </Box>
            )}
          </Grid>
        </Grid>
      </Paper>

      {/* Enhanced Lightbox Dialog */}
      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="xl"
        fullWidth
        TransitionComponent={Zoom}
        PaperProps={{
          sx: { 
            bgcolor: "rgba(18, 18, 18, 0.95)", 
            backdropFilter: "blur(10px)",
            boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5)",
            p: 2,
            borderRadius: 2,
            overflow: "hidden"
          },
        }}
      >
        <Box sx={{ position: "relative" }}>
          <IconButton
            onClick={handleClose}
            sx={{ 
              position: "absolute", 
              top: 16, 
              right: 16, 
              color: "#fff",
              bgcolor: "rgba(0, 0, 0, 0.5)",
              zIndex: 10,
              "&:hover": {
                bgcolor: "rgba(0, 0, 0, 0.7)",
              }
            }}
          >
            <Close />
          </IconButton>
          
          {selected && (
            <>
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" color="white" sx={{ mb: 1 }}>
                  {selected.name}
                </Typography>
                <Typography variant="body2" color="grey.400">
                  {selected.desc} • {selected.category}
                </Typography>
              </Box>
              
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  width: "100%",
                  position: "relative"
                }}
              >
                <Box
                  component="img"
                  src={selected.src}
                  alt={selected.name}
                  sx={{
                    maxWidth: "100%",
                    maxHeight: "75vh",
                    objectFit: "contain",
                    borderRadius: 1,
                    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.2)",
                  }}
                />
              </Box>
              
              <Stack 
                direction="row" 
                justifyContent="flex-end" 
                spacing={1} 
                sx={{ mt: 3 }}
              >
                <Button
                  startIcon={<Download />}
                  variant="contained"
                  onClick={(e) => handleDownload(e, selected)}
                  sx={{
                    bgcolor: "rgba(255, 255, 255, 0.1)",
                    color: "white",
                    "&:hover": {
                      bgcolor: "rgba(255, 255, 255, 0.2)",
                    },
                  }}
                >
                  Download
                </Button>
              </Stack>
            </>
          )}
        </Box>
      </Dialog>
    </Container>
  );
}