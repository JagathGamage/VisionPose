import { Box, Grid, Card, CardContent, Typography } from "@mui/material";
import LEFT from "../graphs/LEFT.png";
import MIDDLE from "../graphs/MIDDLE.png";
import RIGHT from "../graphs/RIGHT.png";
import COMBINED from "../graphs/COMBINED.png";

// Define local paths and their corresponding filenames
const graphImages = [
  { src: LEFT, name: "LEFT.png" },
  { src: MIDDLE, name: "MIDDLE.png" },
  { src: RIGHT, name: "RIGHT.png" },
  { src: COMBINED, name: "COMBINED.png" },
];

export default function GraphDashboard() {
  return (
    <Box sx={{ maxWidth: "1200px", mx: "auto", mt: 4, p: 3 }}>
      <Typography variant="h5" fontWeight="bold" align="center" gutterBottom>
        Motion Analysis Graphs
      </Typography>

      <Grid container spacing={4}>
        {graphImages.map((graph, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card sx={{ boxShadow: 3 }}>
              <CardContent>
                <Typography variant="h6" align="center">
                  {graph.name} {/* Displaying actual file name */}
                </Typography>
                <img
                  src={graph.src}
                  alt={graph.name}
                  style={{
                    width: "100%",
                    borderRadius: "8px",
                    marginTop: "10px",
                    border: "2px solid #ddd",
                  }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
