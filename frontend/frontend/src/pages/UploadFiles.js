import React, { useState } from "react";
import { Button, Box, Paper, Grid, Typography, FormControl, Input } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

function UploadSection({ withDropdown = false, id }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = () => {
    if (selectedFile) {
      console.log("Uploading file:", selectedFile.name);
      // Implement file upload logic here (e.g., API call with FormData)
    }
  };

  return (
    <Paper elevation={10} style={{ padding: 40, height: "40vh", width: 350, marginBottom: "45px" }}>
      <Typography align="center" variant="h6">
        Add Files
      </Typography>

      {withDropdown && (
        <Grid align="center">
          <FormControl sx={{ m: 2, minWidth: 150 }} size="small">
            <Input placeholder="Type angle" variant="outlined" color="primary" />
          </FormControl>
        </Grid>
      )}

      <Box component="section" sx={{ p: 6, border: "1px dashed blue", textAlign: "center" }}>
        <input type="file" onChange={handleFileChange} style={{ display: "none" }} id={`file-input-${id}`} />
        <label htmlFor={`file-input-${id}`}>
          <Button component="span" variant="contained" startIcon={<CloudUploadIcon />}>
            Choose File
          </Button>
        </label>
        {selectedFile && <Typography mt={2}>{selectedFile.name}</Typography>}
      </Box>

      <Box display="flex" justifyContent="center" mt={2}>
        <Button onClick={handleUpload} variant="contained" disabled={!selectedFile}>
          Upload
        </Button>
      </Box>
    </Paper>
  );
}

function UploadFiles() {
  return (
    <>
      <Grid container spacing={10} justifyContent="center">
        <Grid item><UploadSection withDropdown id="1" /></Grid>
        <Grid item><UploadSection withDropdown id="2" /></Grid>
      </Grid>
      <Grid container spacing={10} justifyContent="center">
        <Grid item><UploadSection withDropdown id="3" /></Grid>
      </Grid>
    </>
  );
}

export default UploadFiles;



