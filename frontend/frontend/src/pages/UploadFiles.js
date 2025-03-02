import React, { useState } from "react";
import { Button, Box, Paper, Grid, Typography, FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

function UploadSection({ withDropdown = false, id }) {
  const [selection, setSelection] = useState("");

  return (
    <Paper elevation={10} style={{ padding: 40, height: "35vh", width: 350, marginBottom: "45" }}>
      <Typography align="center" variant="h6">
        Upload Your Files
      </Typography>

      {withDropdown && (
        <Grid align="center">
          <FormControl sx={{ m: 2, minWidth: 150 }} size="small">
            <InputLabel id={`select-label-${id}`}>Select angle</InputLabel>
            <Select
              labelId={`select-label-${id}`}
              id={`select-${id}`}
              value={selection}
              onChange={(event) => setSelection(event.target.value)}
            >
              <MenuItem value=""><em>None</em></MenuItem>
              <MenuItem value={10}>Ten</MenuItem>
              <MenuItem value={20}>Twenty</MenuItem>
              <MenuItem value={30}>Thirty</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      )}

      <Box component="section" sx={{ p: 6, border: "1px dashed blue" }}>
        <Box display="flex" justifyContent="center" mt={2}>
          <Button component="label" variant="contained" startIcon={<CloudUploadIcon />}>
            Upload files
          </Button>
        </Box>
      </Box>
    </Paper>
  );
}

function UploadFiles() {
  return (
    <>
    <Grid container spacing={10} justifyContent="center" >
      
      <Grid item><UploadSection withDropdown id="1" /></Grid>
      <Grid item><UploadSection withDropdown id="2" /></Grid>
   
    </Grid>
     <Grid container spacing={10} justifyContent="center" >
      
     
     <Grid item><UploadSection withDropdown id="3" /></Grid>
     <Grid item><UploadSection withDropdown id="4" /></Grid>
   </Grid>
   </>
  );
}

export default UploadFiles;


