// src/Components/CheatingDetection.jsx
import * as faceapi from 'face-api.js';

// --- Head Pose (Yaw, Pitch, Roll) Estimation Helpers ---
// This is a simplified head pose check based on landmarks.
// It assumes a frontal-facing camera. Thresholds are empirical.
// Yaw: Left/Right, Pitch: Up/Down.

const YAW_THRESHOLD = 25; // degrees
const PITCH_THRESHOLD = 25; // degrees

/**
 * Loads the required face-api.js models from the public/models directory.
 */
export async function loadModels() {
  const MODEL_URL = '/models'
  // Use TinyFaceDetector for speed, and faceLandmarkNet for pose estimation.
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  // Add more as needed: await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  console.log("Face API Models Loaded.");
}


/**
 * Core function for real-time face analysis.
 * @param {HTMLVideoElement} videoEl - The video element showing the webcam stream.
 * @returns {number} The number of faces detected (or -1 if detection fails).
 */
async function runFaceDetection(videoEl) {
  if (!videoEl || videoEl.paused || videoEl.ended || !faceapi.nets.tinyFaceDetector.isLoaded) {
    return -1;
  }

  try {
    const detections = await faceapi
      .detectAllFaces(videoEl, new faceapi.TinyFaceDetectorOptions({ inputSize: 360})) // Smaller input for speed
      .withFaceLandmarks();

    return detections;
  } catch (error) {
    console.error("Face detection error:", error);
    return -1;
  }
}

// --- Exported Helper Functions for VirtualInterviewerStarter ---

/**
 * 1. Multiple Faces Detection & No Face Detection
 */
export async function detectMultipleFaces(videoEl) {
  const detections = await runFaceDetection(videoEl);

  if (detections === -1) return 0; // Return 0 if function failed or not running

  if (detections.length > 1) {
    return detections.length; // Will be > 1
  }
  
  // No face detected for the purpose of the initial check
  return detections.length; // Will be 0 or 1
}

/**
 * 2. Grazing Away Detection (Head Pose)
 */
export async function ensureFacePresentAndForward(videoEl, warningCallback) {
  const detections = await runFaceDetection(videoEl);

  if (detections === -1) return;

  if (detections.length === 0) {
    warningCallback("No face detected.");
    return;
  }
  
  // Assume one main face for now. Analyze head pose.
  const detection = detections[0]; 
  
  // Face-api.js doesn't provide easy Yaw/Pitch/Roll, but we can approximate
  // using landmark geometry (e.g., nose tip position relative to the face box/eyes).
  
  // A simple approximation: check if the face box aspect ratio is too distorted 
  // or if the center of the face landmarks is too far from the center of the box.
  
  // More accurate: calculate 3D head pose using face landmarks.
  // For a full implementation, you'd need a PnP solver, which is complex for a snippet.
  // We'll use a simplified check based on eye/nose position relative to the box center.
  
  const box = detection.detection.box;
  const landmarks = detection.landmarks;
  
  // Simplified Yaw approximation: check horizontal center of the nose tip
  const noseTip = landmarks.getNose()[3];
  const boxCenter = box.x + box.width / 2;
  const horizontalDistance = Math.abs(noseTip.x - boxCenter);
  const yawDeviation = (horizontalDistance / box.width) * 100; // % deviation

  // Simplified Pitch approximation: check vertical center of the chin/jaw
  const chin = landmarks.getJawOutline()[8];
  const boxHeight = box.height;
  const verticalDistance = chin.y - (box.y + boxHeight);
  const pitchDeviation = (verticalDistance / boxHeight) * 100;

  // Set rough thresholds for cheating (looking too far to the side/down)
  if (yawDeviation > 25) { // e.g., nose tip is more than 25% of the box width away from the center
    warningCallback("Looking away from the screen.");
    return;
  }
  
  if (pitchDeviation > 10) { // e.g., chin is too far below expected position (looking down)
      // This pitch check is very crude and may be better omitted or replaced with proper pose math.
      // warningCallback("Looking down, possibly reading.");
      // return;
  }
  
  warningCallback(null); // Clear warning if detection is successful and face is present/forward
}

/**
 * 3. Tab Switching/Focus Detection
 */
export function monitorTabSwitch(callback) {
  const handleVisibilityChange = () => {
    if (document.hidden) {
      callback("Tab switched or minimized");
    }
  };

  const handleWindowBlur = () => {
    // This catches when the user clicks out of the browser window entirely
    if (!document.hidden) { 
        callback("Window focus lost (switched to another application)");
    }
  };

  document.addEventListener('visibilitychange', handleVisibilityChange);
  window.addEventListener('blur', handleWindowBlur);

  // Return a cleanup function
  return () => {
    document.removeEventListener('visibilitychange', handleVisibilityChange);
    window.removeEventListener('blur', handleWindowBlur);
  };
}

// Export a dummy component, as its functions are used elsewhere
export default function CheatingDetection() {
  return null;
}