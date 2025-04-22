import { useState } from "react";
import { Card, CardContent, Typography, Button, Grid } from "@mui/material";

const requirements = [
  "Full Bowling Action Analysis",
  "Arm Angle Measurement",
  "Run-up Distance & Speed Calculation",
  "Front Foot Positioning",
  "Back Foot Contact Analysis",
  "Hip & Shoulder Alignment",
  "Release Point Height & Angle",
  "Follow-through Motion Analysis",
  "Knee Flexion & Stability Check",
  "Custom Angle & Distance Measurement",
];

export default function RequirementSelector() {
  const [selectedRequirement, setSelectedRequirement] = useState("");

  return (
    <Card sx={{ maxWidth: 600, mx: "auto", p: 3, boxShadow: 3, borderRadius: 2 }}>
      {/* Heading */}
      <Typography variant="h5" fontWeight="bold" align="center" gutterBottom>
        Select an Analysis Requirement
      </Typography>
      <Typography variant="body2" color="textSecondary" align="center" mb={2}>
        Choose an option to analyze specific aspects of the bowling action.
      </Typography>

      {/* Requirement Buttons */}
      <Grid container spacing={2}>
        {requirements.map((req, index) => (
          <Grid item xs={12} sm={6} key={index}>
            <Button
              fullWidth
              variant={selectedRequirement === req ? "contained" : "outlined"}
              color={selectedRequirement === req ? "primary" : "inherit"}
              onClick={() => setSelectedRequirement(req)}
              sx={{
                textTransform: "none",
                fontSize: "0.9rem",
                borderRadius: 2,
                py: 1.5,
              }}
            >
              {req}
            </Button>
          </Grid>
        ))}
      </Grid>

      {/* Display Selected Requirement */}
      {selectedRequirement && (
        <Card sx={{ mt: 3, p: 2, bgcolor: "primary.light", color: "white" }}>
          <Typography variant="h6" align="center">
            Selected Requirement:
          </Typography>
          <Typography variant="body1" align="center" fontWeight="bold">
            {selectedRequirement}
          </Typography>
        </Card>
      )}
    </Card>
  );
}
