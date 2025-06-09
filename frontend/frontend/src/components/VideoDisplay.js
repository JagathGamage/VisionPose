import React from "react";
import { Box, Typography, Card, CardContent, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";

const VideoDisplay = () => {
  const navigate = useNavigate();

  const videoUrls = [
    "http://127.0.0.1:8000/output/right-shoulder-angles-sample-b.mp4",
    "http://127.0.0.1:8000/output/right-shoulder-angles-sample-b.mp4",
    "http://127.0.0.1:8000/output/right-shoulder-angles-sample-c.mp4",
  ];

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#f5f5f5", p: 4 }}>
      <Typography variant="h4" align="center" gutterBottom fontWeight={700}>
        Videos with angle variations
      </Typography>

      <Box
        sx={{
          display: "flex",
          gap: 3,
          flexDirection: { xs: "column", md: "row" },
          justifyContent: "center",
          alignItems: "stretch",
          mt: 4,
        }}
      >
        {videoUrls.map((url, index) => (
          <Card
            key={index}
            sx={{
              maxWidth: 400,
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              p: 2,
              boxShadow: 4,
              borderRadius: 3,
            }}
          >
            <CardContent sx={{ width: "100%", textAlign: "center" }}>
              <Typography variant="h6" gutterBottom>
                Video {index + 1}
              </Typography>
              <Box
                component="video"
                src={url}
                controls
                sx={{
                  width: "100%",
                  height: 250,
                  objectFit: "cover",
                  borderRadius: 2,
                }}
              />
            </CardContent>
          </Card>
        ))}
      </Box>

      {/* Button to go to graphs page */}
      <Box sx={{ display: "flex", justifyContent: "center", mt: 6 }}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => navigate("/graphs")}
          sx={{
            px: 4,
            py: 1.5,
            borderRadius: 2,
            fontWeight: "bold",
            textTransform: "none",
            boxShadow: 3,
          }}
        >
          View Graphs
        </Button>
      </Box>
    </Box>
  );
};

export default VideoDisplay;
