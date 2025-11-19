// src/main.jsx
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./index.css";

import VirtualInterviewerStarter from "./Components/VirtualInterviewerStarter";
import EmployerDashboard from "./Components/EmployerDashboard";
import LandingPage from "./Components/landingpage"; // ensure filename & export match exactly

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/candidate" element={<VirtualInterviewerStarter />} />
        <Route path="/employer" element={<EmployerDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

// Mount the app
const container = document.getElementById("root");
if (!container) {
  // helpful guard while debugging — will print if index.html is missing #root
  throw new Error("Root container (#root) not found — make sure public/index.html has <div id='root'></div>");
}
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
