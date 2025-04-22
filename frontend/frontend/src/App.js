import React from 'react';
import "./App.css";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import {Login,Signup} from "./Routes.js";

import UploadFiles from "./pages/UploadFiles.js";
import Trimvideos from "./pages/trimvideos.js";

import RequirementSelector from './components/RequirementSelector.js';
import VideoTrimmer from './components/VideoTrimmer.js';
import GraphDashboard from './components/Graphs.js';
import VideoUpload from './components/VideoUpload.js';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/Login' element={<Login />} />
        <Route path='/Signup' element={<Signup />} />

        <Route path='/UploadFiles' element={<VideoUpload />} />

        <Route path='/Trimvideos' element={<Trimvideos />} />
        trimvideos


        <Route path='/requirementSelector' element={<RequirementSelector />} />
        <Route path='/videoTrimmer' element={<VideoTrimmer />} />
        <Route path='/graphs' element={<GraphDashboard />} />


      </Routes>
    </BrowserRouter>
  );
}

export default App;
