import React from 'react';
import "./App.css";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import {Login,Signup} from "./Routes.js";

import UploadFiles from "./pages/UploadFiles.js";

import RequirementSelector from './components/RequirementSelector.js';
import VideoTrimmer from './components/VideoTrimmer.js';
import GraphDashboard from './components/Graphs.js';
import VideoUpload from './components/VideoUpload.js';
import Sync from './components/Sync.js';
import FinalVideoPage from './components/FinalVideoPage.js';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Login />} />
        <Route path='/Signup' element={<Signup />} />

        <Route path='/UploadFiles' element={<VideoUpload />} />
        <Route path='/sync' element={<Sync />} />

        <Route path='/requirementSelector' element={<RequirementSelector />} />
        <Route path='/videoTrimmer' element={<VideoTrimmer />} />
        <Route path='/graphs' element={<GraphDashboard />} />
        <Route path="/final-video" element={<FinalVideoPage />} />

      </Routes>
    </BrowserRouter>
  );
}

export default App;
