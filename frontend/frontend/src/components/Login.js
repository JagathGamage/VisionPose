import React, { useState } from 'react';
import {
  TextField,
  Button,
  FormControlLabel,
  Checkbox,
  Paper,
  Avatar,
  Typography,
  Link,
  Grid
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const paperStyle = {
    padding: 20,
    height: '70vh',
    width: 400,
    margin: '20px auto'
  };
  const avatarStyle = { backgroundColor: '#3f51b5' };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (email === 'admin@gmail.com' && password === '1234') {
      navigate('/UploadFiles');
    } else {
      alert('Invalid credentials');
    }
  };

  return (
    <Grid container justifyContent="center" alignItems="center">
      <Paper elevation={10} style={paperStyle}>
        <Grid container direction="column" alignItems="center">
          <Avatar style={avatarStyle}>
            <LockOutlinedIcon />
          </Avatar>
          <Typography variant="h5" gutterBottom>
            Sign In
          </Typography>
        </Grid>
        <form onSubmit={handleSubmit}>
          <TextField
            label="User Name"
            variant="outlined"
            size="small"
            fullWidth
            required
            margin="normal"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <TextField
            label="Password"
            type="password"
            variant="outlined"
            size="small"
            fullWidth
            required
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Remember me"
          />
          <Button
            type="submit"
            color="primary"
            variant="contained"
            fullWidth
            sx={{ mt: 2 }}
          >
            Sign In
          </Button>
        </form>
        <Typography sx={{ mt: 2 }}>
          <Link href="#">Forgot password?</Link>
        </Typography>
        <Typography sx={{ mt: 2 }}>
          Don't have an account? <Link href="#">Sign Up</Link>
        </Typography>
      </Paper>
    </Grid>
  );
}

export default Login;
