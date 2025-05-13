import { useState } from "react";
import { Card, CardContent, Typography, Button, Grid, Box } from "@mui/material";
import { useNavigate } from "react-router-dom";

const requirements = [
  " Duration of each of the last five steps in the run-up and the delivery stride ",
  "Delivery stride duration and length (i.e., from back foot contact to front foot contact) ",
  "Bowling Elbow flexion-extension angle from bowling arm horizontally behind the body to the instant of ball release",
  "Front Knee flexion-extension angle from back foot contact to front foot contact ",
  "Front Knee flexion-extension angle from front foot contact to ball release ",
  "Thorax and pelvis: flexion, lateral bending/obliquity, and rotation during the pre-delivery jump phase  ",
  "Thorax and pelvis: flexion, lateral bending/obliquity, and rotation from back foot contact to front foot contact ",
  "Thorax and pelvis: flexion, lateral bending/obliquity, and rotation from front foot contact to ball release",
  "Centre of mass of head x, y, z distance with respect to the front foot during the entire front foot contact phase",
  " Wrist joint speed (x, y, z) and Ball release speed from front foot contact to the instant of ball release",
];

export default function RequirementSelector() {
  const [selectedRequirement, setSelectedRequirement] = useState("");
  const navigate = useNavigate();

  const handleNext = () => {
    navigate("/videoTrimmer", { state: { selectedRequirement } });
  };

  return (
    <Card sx={{ maxWidth: 600, mx: "auto", p: 3, mt: 5, boxShadow: 3, borderRadius: 2 }}>
      {/* Heading */}
      <Typography variant="h5" fontWeight="bold" align="center" gutterBottom>
        Select an Analysis Requirement
      </Typography>
      <Typography variant="body2" color="textSecondary" align="center" mb={2}>
        Choose an option to analyze specific aspects of the bowling action.
      </Typography>

      {/* Scrollable List of Requirements */}
      <Box sx={{ maxHeight: 500, overflowY: "auto", pr: 1 }}>
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
      </Box>

      {/* Display Selected Requirement */}
      {selectedRequirement && (
        <>
          <Card sx={{ mt: 3, p: 2, bgcolor: "primary.light", color: "white" }}>
            <Typography variant="h6" align="center">
              Selected Requirement:
            </Typography>
            <Typography variant="body1" align="center" fontWeight="bold">
              {selectedRequirement}
            </Typography>
          </Card>

          {/* Next Button */}
          <Button
            fullWidth
            variant="contained"
            color="primary"
            sx={{ mt: 3, py: 1.5, borderRadius: 2 }}
            onClick={handleNext}
          >
            Next
          </Button>
        </>
      )}
    </Card>
  );
}
