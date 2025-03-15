import React from 'react';
import "./App.css";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import {Login,Signup} from "./Routes.js";
import RequirementSelector from './components/RequirementSelector.js';
import VideoTrimmer from './components/VideoTrimmer.js';
import GraphDashboard from './components/Graphs.js';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/Login' element={<Login />} />
        <Route path='/Signup' element={<Signup />} />
        <Route path='/requirementSelector' element={<RequirementSelector />} />
        <Route path='/videoTrimmer' element={<VideoTrimmer />} />
        <Route path='/graphs' element={<GraphDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
