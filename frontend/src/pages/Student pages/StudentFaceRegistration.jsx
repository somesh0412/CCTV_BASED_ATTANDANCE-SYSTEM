import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CheckCircle, LoaderCircle, RotateCcw, XCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import api from "../../api/axious";
import StudentSidebar from "./StudentSidebar";
import "./StudentFaceRegistration.css";

const FRAME_INTERVAL_MS = 700;
const TARGET_SAMPLES = 30;
const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled", "timed_out"]);

function getApiMessage(error, fallback) {
  if (!error.response) {
    return "Unable to reach the server. Check your connection and try again.";
  }
  return error.response.data?.message || fallback;
}

function getStoredStudent() {
  try {
    return JSON.parse(localStorage.getItem("student") || "null");
  } catch {
    return null;
  }
}

export default function StudentFaceRegistration() {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const sessionIdRef = useRef("");
  const loopTimerRef = useRef(null);
  const isMountedRef = useRef(true);
  const isRunningRef = useRef(false);
  const frameInFlightRef = useRef(false);
  const lastFrameSignatureRef = useRef("");
  const captureAndUploadFrameRef = useRef(null);

  const [progress, setProgress] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [cameraError, setCameraError] = useState("");
  const [lastRejectionReason, setLastRejectionReason] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopLoop = useCallback(() => {
    isRunningRef.current = false;
    setIsRunning(false);
    if (loopTimerRef.current) {
      clearTimeout(loopTimerRef.current);
      loopTimerRef.current = null;
    }
  }, []);

  const finishEnrollment = useCallback(() => {
    stopLoop();
    stopCamera();
  }, [stopCamera, stopLoop]);

  const handleCompleted = useCallback(() => {
    finishEnrollment();
    const student = getStoredStudent();
    if (student) {
      localStorage.setItem(
        "student",
        JSON.stringify({ ...student, faceRegistered: true })
      );
    }
    navigate("/student-dashboard", { replace: true });
  }, [finishEnrollment, navigate]);

  const applyProgress = useCallback(
    (nextProgress) => {
      if (!nextProgress) {
        return false;
      }

      setProgress(nextProgress);
      const nextStatus = nextProgress.current_status || "active";
      setStatus(nextStatus);
      if (nextStatus === "completed") {
        handleCompleted();
        return true;
      }
      if (TERMINAL_STATUSES.has(nextStatus)) {
        finishEnrollment();
        return true;
      }
      return false;
    },
    [finishEnrollment, handleCompleted]
  );

  const scheduleNextFrame = useCallback(() => {
    if (isRunningRef.current && captureAndUploadFrameRef.current) {
      loopTimerRef.current = setTimeout(
        captureAndUploadFrameRef.current,
        FRAME_INTERVAL_MS
      );
    }
  }, []);

  const captureAndUploadFrame = useCallback(async () => {
    if (
      !isRunningRef.current ||
      frameInFlightRef.current ||
      !videoRef.current ||
      videoRef.current.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      scheduleNextFrame();
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!canvas || video.videoWidth === 0 || video.videoHeight === 0) {
      scheduleNextFrame();
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = 16;
    sampleCanvas.height = 12;
    const sampleContext = sampleCanvas.getContext("2d");
    sampleContext.drawImage(video, 0, 0, 16, 12);
    const signature = Array.from(
      sampleContext.getImageData(0, 0, 16, 12).data
    ).join(",");
    if (signature === lastFrameSignatureRef.current) {
      scheduleNextFrame();
      return;
    }
    lastFrameSignatureRef.current = signature;

    frameInFlightRef.current = true;
    try {
      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.85);
      });
      if (!blob || !isRunningRef.current) {
        return;
      }

      const formData = new FormData();
      formData.append("session_id", sessionIdRef.current);
      formData.append("frame", blob, "enrollment-frame.jpg");
      const response = await api.post(
        "/student-enrollment/process-frame",
        formData
      );
      if (!isMountedRef.current) {
        return;
      }

      const responseData = response.data?.data || {};
      if (responseData.progress) {
        applyProgress(responseData.progress);
      }
      if (responseData.reason) {
        setLastRejectionReason(responseData.reason);
      }
    } catch (requestError) {
      if (isMountedRef.current && isRunningRef.current) {
        finishEnrollment();
        setStatus("failed");
        setError(getApiMessage(requestError, "Frame upload failed."));
      }
    } finally {
      frameInFlightRef.current = false;
      scheduleNextFrame();
    }
  }, [applyProgress, finishEnrollment, scheduleNextFrame]);

  useEffect(() => {
    captureAndUploadFrameRef.current = captureAndUploadFrame;
  }, [captureAndUploadFrame]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      if (!isMountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setIsCameraActive(true);
      setCameraError("");
      return true;
    } catch {
      setCameraError(
        "Camera access was denied or unavailable. Allow webcam access and try again."
      );
      return false;
    }
  }, []);

  const startEnrollment = useCallback(async () => {
    setError("");
    setCameraError("");
    setLastRejectionReason("");
    setIsStarting(true);
    stopLoop();
    stopCamera();

    try {
      const response = await api.post("/student-enrollment/start");
      const nextSessionId = response.data?.data?.flaskSessionId;
      const startProgress = response.data?.data?.progress;
      if (!nextSessionId || !startProgress) {
        throw new Error("The enrollment session response was incomplete.");
      }

      sessionIdRef.current = nextSessionId;
      setProgress(startProgress);
      setStatus(startProgress.current_status || "active");

      if (!(await startCamera())) {
        await api.post(
          `/student-enrollment/cancel/${encodeURIComponent(nextSessionId)}`
        );
        setStatus("cancelled");
        return;
      }

      isRunningRef.current = true;
      setIsRunning(true);
      scheduleNextFrame();
    } catch (requestError) {
      setStatus("failed");
      setError(
        requestError.message === "The enrollment session response was incomplete."
          ? requestError.message
          : getApiMessage(requestError, "Could not start face enrollment.")
      );
    } finally {
      setIsStarting(false);
    }
  }, [scheduleNextFrame, startCamera, stopCamera, stopLoop]);

  const cancelEnrollment = useCallback(async () => {
    stopLoop();
    stopCamera();
    if (!sessionIdRef.current) {
      setStatus("cancelled");
      return;
    }

    try {
      const response = await api.post(
        `/student-enrollment/cancel/${encodeURIComponent(sessionIdRef.current)}`
      );
      const responseProgress = response.data?.data?.progress;
      if (responseProgress) {
        setProgress(responseProgress);
      }
      setStatus("cancelled");
    } catch (requestError) {
      setError(getApiMessage(requestError, "Could not cancel enrollment."));
      setStatus("failed");
    }
  }, [stopCamera, stopLoop]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      stopLoop();
      stopCamera();
    };
  }, [stopCamera, stopLoop]);

  const acceptedSamples = progress?.accepted_sample_count || 0;
  const percentage = Math.min(
    100,
    Math.round((acceptedSamples / TARGET_SAMPLES) * 100)
  );
  const statusLabel = status.replaceAll("_", " ");

  return (
    <div className="student-enrollment-page">
      <StudentSidebar activeItem="face-registration" />
      <main className="student-enrollment-content">
        <section className="student-enrollment-card">
          <div className="student-enrollment-heading">
            <div>
              <p className="student-enrollment-eyebrow">Student portal</p>
              <h1>Face Registration</h1>
              <p>
                Keep your face centered and slowly change your angle when
                prompted.
              </p>
            </div>
            <Camera size={32} aria-hidden="true" />
          </div>

          {error && <div className="enrollment-alert error">{error}</div>}
          {cameraError && (
            <div className="enrollment-alert error">{cameraError}</div>
          )}

          <div className="enrollment-camera-panel">
            <video
              ref={videoRef}
              className="enrollment-video"
              muted
              playsInline
              aria-label="Live webcam preview"
            />
            {!isCameraActive && (
              <div className="enrollment-camera-placeholder">
                <Camera size={40} />
                <span>Camera preview will appear here</span>
              </div>
            )}
          </div>
          <canvas ref={canvasRef} className="enrollment-canvas" />

          <div className="enrollment-progress-header">
            <strong>
              Accepted samples: {acceptedSamples} / {TARGET_SAMPLES}
            </strong>
            <span>{percentage}%</span>
          </div>
          <div className="enrollment-progress-track">
            <div
              className="enrollment-progress-value"
              style={{ width: `${percentage}%` }}
            />
          </div>

          <div className="enrollment-details">
            <div>
              <span>Status</span>
              <strong>{statusLabel}</strong>
            </div>
            <div>
              <span>Instruction</span>
              <strong>
                {progress?.suggested_student_instruction ||
                  "Start enrollment to receive camera instructions."}
              </strong>
            </div>
            <div>
              <span>Last rejection reason</span>
              <strong>{lastRejectionReason || "None"}</strong>
            </div>
          </div>

          <div className="enrollment-actions">
            {!isRunning && (
              <button
                type="button"
                className="enrollment-button primary"
                onClick={startEnrollment}
                disabled={isStarting}
              >
                {isStarting ? (
                  <>
                    <LoaderCircle className="spin" size={18} />
                    Starting...
                  </>
                ) : status === "idle" ? (
                  "Start enrollment"
                ) : (
                  <>
                    <RotateCcw size={18} />
                    Retry
                  </>
                )}
              </button>
            )}
            {isRunning && (
              <button
                type="button"
                className="enrollment-button secondary"
                onClick={cancelEnrollment}
              >
                <XCircle size={18} />
                Cancel
              </button>
            )}
          </div>

          {status === "completed" && (
            <div className="enrollment-success">
              <CheckCircle size={20} />
              Enrollment complete. Redirecting to your dashboard...
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
