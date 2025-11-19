// src/Components/landingpage.jsx
import React from "react";
import { Link } from "react-router-dom";
import Logo from "../assets/logo5.png";            // make sure this path is correct
 // optional - if you don't have it, delete this import

export default function LandingPage() {
  // Use backgroundUrl only if BackgroundImage exists. If you removed the import, set to empty string.
  const backgroundUrl = typeof BackgroundImage === "string" ? `url(${BackgroundImage})` : "";

  return (
    <div
    className="min-h-screen bg-cover bg-center p-6"
    style={{ backgroundImage: "url('/bg3.jpg')" }}
    >

      {/* Header with logo */}
      <header className="container mx-auto px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Logo import must exist at src/assets/logo.svg */}
          <img src={Logo} alt="logo" className="w-12 h-12 object-contain" />
          <div>
            <h1 className="text-lg font-semibold text-slate-800">AIInterviewer</h1>
            <p className="text-sm text-slate-500">Virtual interviews with AI feedback</p>
          </div>
        </div>

        <nav className="hidden md:flex gap-6 items-center text-slate-600">
          <a className="hover:text-slate-800" href="#features">
            Features
          </a>
          <a className="hover:text-slate-800" href="#how">
            How
          </a>
          <a className="hover:text-slate-800" href="#contact">
            Contact
          </a>
        </nav>
      </header>

      {/* Hero */}
      <main className="flex-grow container mx-auto px-6 py-12 flex flex-col items-center">
        <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4 text-center">
         Assess. Analyze. Advance..
        </h2>
        <p className="text-center text-slate-600 max-w-xl">
          Structured Virtual interviews with AI-driven feedback on communication and technical performance.
        </p>

        {/* Side-by-side small squares (clickable) */}
        <div className="mt-10 grid grid-cols-2 gap-6 max-w-md w-full">
          <Link
            to="/candidate"
            className="group h-36 rounded-xl bg-white shadow-md flex flex-col items-center justify-center hover:shadow-lg transition cursor-pointer"
          >
            <div className="w-12 h-12 rounded-lg bg-indigo-100 flex items-center justify-center mb-2">
              <span className="text-indigo-600 text-xl" aria-hidden>
                👤
              </span>
            </div>
            <h3 className="text-lg font-semibold text-slate-800 group-hover:text-indigo-600">Candidate</h3>
          </Link>

          <Link
            to="/employer"
            className="group h-36 rounded-xl bg-white shadow-md flex flex-col items-center justify-center hover:shadow-lg transition cursor-pointer"
          >
            <div className="w-12 h-12 rounded-lg bg-violet-100 flex items-center justify-center mb-2">
              <span className="text-violet-600 text-xl" aria-hidden>
                💼
              </span>
            </div>
            <h3 className="text-lg font-semibold text-slate-800 group-hover:text-violet-600">Employer</h3>
          </Link>
        </div>

        {/* Features section */}
        <section id="features" className="mt-12 w-full max-w-4xl">
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div className="p-4 bg-white rounded-lg shadow-sm">
              <h5 className="font-medium">AI feedback</h5>
              <p className="text-sm text-slate-500 mt-2">Automated scoring on clarity and confidence.</p>
            </div>
            <div className="p-4 bg-white rounded-lg shadow-sm">
              <h5 className="font-medium">Templates</h5>
              <p className="text-sm text-slate-500 mt-2">Role-specific interview templates and rubrics.</p>
            </div>
            <div className="p-4 bg-white rounded-lg shadow-sm">
              <h5 className="font-medium">Secure recordings</h5>
              <p className="text-sm text-slate-500 mt-2">Encrypted candidate videos and shareable reports.</p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer id="contact" className="border-t py-6 bg-white">
        <div className="container mx-auto px-6 text-sm text-slate-500 flex justify-between">
          <div>© {new Date().getFullYear()} InterviewAI</div>
          <div className="flex gap-4">
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
