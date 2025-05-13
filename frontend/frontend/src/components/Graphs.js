import { useState } from "react";
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
} from "@mui/material";
import { ShowChart, Close } from "@mui/icons-material";

import LEFT from "../graphs/LEFT.png";
import MIDDLE from "../graphs/MIDDLE.png";
import RIGHT from "../graphs/RIGHT.png";
import COMBINED from "../graphs/COMBINED.png";

const graphImages = [
  { src: LEFT, name: "LEFT.png", desc: "Left trajectory over time" },
  { src: MIDDLE, name: "MIDDLE.png", desc: "Velocity & acceleration metrics" },
  { src: RIGHT, name: "RIGHT.png", desc: "Comparative right-side patterns" },
  { src: COMBINED, name: "COMBINED.png", desc: "All patterns integrated" },
];

export default function GraphDashboard() {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState(null);

  const handleOpen = (graph) => {
    setSelected(graph);
    setOpen(true);
  };
  const handleClose = () => setOpen(false);

  return (
    <Box sx={{ maxWidth: 1200, mx: "auto", my: 6, px: 2 }}>
      {/* Header */}
      <Stack
        direction="row"
        alignItems="center"
        spacing={1}
        sx={{ mb: 4, justifyContent: "center" }}
      >
        <ShowChart color="primary" fontSize="large" />
        <Typography variant="h4" fontWeight="bold">
          Motion Analysis
        </Typography>
      </Stack>

      {/* Grid of Cards */}
      <Grid container spacing={4}>
        {graphImages.map((graph) => (
          <Grid item xs={12} sm={6} md={3} key={graph.name}>
            <Card
              sx={{
                transition: "transform 0.3s, box-shadow 0.3s",
                "&:hover": {
                  transform: "scale(1.03)",
                  boxShadow: 6,
                },
              }}
            >
              <CardActionArea onClick={() => handleOpen(graph)}>
                <CardMedia
                  component="img"
                  image={graph.src}
                  alt={graph.name}
                  sx={{
                    height: 180,
                    objectFit: "cover",
                  }}
                />
                <CardContent>
                  <Typography
                    variant="subtitle1"
                    fontWeight="600"
                    noWrap
                    gutterBottom
                  >
                    {graph.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" noWrap>
                    {graph.desc}
                  </Typography>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Lightbox Dialog */}
      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { position: "relative", p: 1, background: "transparent", boxShadow: "none" },
        }}
      >
        <IconButton
          onClick={handleClose}
          sx={{ position: "absolute", top: 8, right: 8, color: "#fff", zIndex: 1 }}
        >
          <Close fontSize="large" />
        </IconButton>
        {selected && (
          <Box
            component="img"
            src={selected.src}
            alt={selected.name}
            sx={{
              width: "100%",
              height: "auto",
              borderRadius: 2,
              boxShadow: 3,
            }}
          />
        )}
      </Dialog>
    </Box>
  );
}
