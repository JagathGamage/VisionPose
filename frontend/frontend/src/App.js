import React from 'react';
import "./App.css";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import {Login,Signup} from "./Routes.js";
import UploadFiles from "./pages/UploadFiles.js";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/Login' element={<Login />} />
        <Route path='/Signup' element={<Signup />} />
        <Route path='/UploadFiles' element={<UploadFiles />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
