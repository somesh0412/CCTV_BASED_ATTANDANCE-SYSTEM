import React, { useEffect, useState } from "react";
import {
  Video,
  Camera,
  Users,
  UserCheck,
  UserX,
  PlayCircle,
  StopCircle,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import "./LiveAttendance.css";

/*
  =========================================================
  LIVE ATTENDANCE CONFIGURATION
  =========================================================
*/

const FACE_ENGINE_URL = "http://localhost:8000";
const ATTENDANCE_API_URL = "http://localhost:5000/api/attendance";

const SESSION_INFO = {
  classDivision: "TY CSD - D",
  subject: "Machine Learning",
  room: "201",
  time: "10:15 AM – 11:15 AM",
};

const PIPELINE_STEPS = [
  "CCTV",
  "Face Detection",
  "Tracking",
  "Face Recognition",
  "Attendance",
];


/*
  =========================================================
  LIVE ATTENDANCE COMPONENT
  =========================================================
*/

export default function LiveAttendance() {
  const [sessionStatus, setSessionStatus] = useState("idle");
  const [attendance, setAttendance] = useState([]);

  const [loadingAttendance, setLoadingAttendance] = useState(false);
  const [apiError, setApiError] = useState("");

  /*
    Number of unknown faces cannot currently be obtained
    from the attendance API because unknown faces are not
    stored as attendance records.
    
    We keep this at 0 for now.
    Later we can add a dedicated live recognition API.
  */
  const [detectedFaces, setDetectedFaces] = useState(0);


  /*
    =======================================================
    GET TODAY'S DATE
    =======================================================
  */

  const getTodayDate = () => {
    const now = new Date();

    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");

    return `${year}-${month}-${day}`;
  };


  /*
    =======================================================
    FETCH ATTENDANCE FROM NODE BACKEND
    =======================================================
  */

  const fetchAttendance = async () => {
    try {
      setLoadingAttendance(true);
      setApiError("");

      const today = getTodayDate();

      /*
        The backend currently supports:

        GET /api/attendance

        If your backend later supports date filtering,
        we can change this to:

        /api/attendance?date=YYYY-MM-DD
      */

      const response = await fetch(
        `${ATTENDANCE_API_URL}?date=${today}`
      );

      if (!response.ok) {
        throw new Error(
          `Attendance API returned ${response.status}`
        );
      }

      const data = await response.json();

      /*
        We support a few possible backend response shapes:

        1. { success: true, attendance: [...] }

        2. { success: true, records: [...] }

        3. { success: true, data: [...] }

        4. [...]
      */

      let records = [];

      if (Array.isArray(data)) {
        records = data;
      } else if (Array.isArray(data.attendance)) {
        records = data.attendance;
      } else if (Array.isArray(data.records)) {
        records = data.records;
      } else if (Array.isArray(data.data)) {
        records = data.data;
      }

      setAttendance(records);

    } catch (error) {
      console.error(
        "Failed to fetch attendance:",
        error
      );

      setApiError(
        "Could not load attendance from backend."
      );

    } finally {
      setLoadingAttendance(false);
    }
  };


  /*
    =======================================================
    INITIAL ATTENDANCE LOAD
    =======================================================
  */

  useEffect(() => {
    fetchAttendance();
  }, []);


  /*
    =======================================================
    REFRESH ATTENDANCE WHILE LIVE
    =======================================================

    The Python face engine sends attendance to Node.js.

    React periodically asks Node.js for the latest records.
  */

  useEffect(() => {
    if (sessionStatus !== "live") {
      return;
    }

    const interval = setInterval(() => {
      fetchAttendance();
    }, 2000);

    return () => {
      clearInterval(interval);
    };
  }, [sessionStatus]);


  /*
    =======================================================
    START LIVE ATTENDANCE
    =======================================================
  */

  const handleStart = async () => {
    setApiError("");

    /*
      Check whether the Python streaming server is running.
    */

    try {
      const response = await fetch(
        `${FACE_ENGINE_URL}/health`
      );

      if (!response.ok) {
        throw new Error("Face engine is not healthy.");
      }

      setSessionStatus("live");

      /*
        Load latest attendance immediately.
      */

      await fetchAttendance();

    } catch (error) {
      console.error(
        "Face engine connection failed:",
        error
      );

      setApiError(
        "Face engine is not running. Start the Python streaming server first."
      );

      setSessionStatus("stopped");
    }
  };


  /*
    =======================================================
    STOP LIVE ATTENDANCE
    =======================================================
  */

  const handleStop = () => {
    setSessionStatus("stopped");
  };


  /*
    =======================================================
    MANUAL REFRESH
    =======================================================
  */

  const handleRefresh = () => {
    fetchAttendance();
  };


  /*
    =======================================================
    ATTENDANCE COUNTS
    =======================================================
  */

  const recognized = attendance.length;

  const unknown = 0;

  /*
    detectedFaces is currently based on attendance records.

    Later, when we expose a real recognition endpoint from
    Python, this can become the actual number of faces
    currently visible in the camera.
  */

  const displayedDetectedFaces =
    detectedFaces > recognized
      ? detectedFaces
      : recognized;


  /*
    =======================================================
    FORMAT TIME
    =======================================================
  */

  const formatTime = (value) => {
    if (!value) return "--";

    /*
      Backend may return:
      "06:30:00"
      or ISO datetime.
    */

    if (value.includes("T")) {
      const date = new Date(value);

      if (Number.isNaN(date.getTime())) {
        return value;
      }

      return date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    }

    return value;
  };


  /*
    =======================================================
    RENDER
    =======================================================
  */

  return (
    <div className="la-page">

      {/* =================================================
          HEADER
      ================================================= */}

      <div className="la-header">

        <div>

          <h1>
            Live Attendance
          </h1>

          <p>
            Monitor the classroom and mark attendance automatically.
          </p>

        </div>


        <span
          className={`la-status la-status--${sessionStatus}`}
        >

          <span className="la-status__dot" />

          {sessionStatus === "live"
            ? "Live"
            : sessionStatus === "stopped"
            ? "Stopped"
            : "Idle"}

        </span>

      </div>


      {/* =================================================
          ERROR MESSAGE
      ================================================= */}

      {apiError && (
        <div
          style={{
            marginBottom: "16px",
            padding: "12px 16px",
            borderRadius: "8px",
            background: "#fff1f2",
            color: "#b42318",
            border: "1px solid #fecdd3",
          }}
        >
          {apiError}
        </div>
      )}


      {/* =================================================
          SESSION INFORMATION
      ================================================= */}

      <div className="la-card la-session-bar">

        <div>

          <span className="la-session-bar__label">
            Class
          </span>

          <strong>
            {SESSION_INFO.classDivision}
          </strong>

        </div>


        <div>

          <span className="la-session-bar__label">
            Subject
          </span>

          <strong>
            {SESSION_INFO.subject}
          </strong>

        </div>


        <div>

          <span className="la-session-bar__label">
            Room
          </span>

          <strong>
            {SESSION_INFO.room}
          </strong>

        </div>


        <div>

          <span className="la-session-bar__label">
            Time
          </span>

          <strong>
            {SESSION_INFO.time}
          </strong>

        </div>

      </div>


      {/* =================================================
          MAIN GRID
      ================================================= */}

      <div className="la-grid">


        {/* =================================================
            VIDEO CARD
        ================================================= */}

        <div className="la-card la-video-card">

          <div className="la-video">

            {sessionStatus === "live" ? (

              /*
                IMPORTANT:

                This is the actual Python Flask stream.

                Python:
                localhost:8000/video_feed
              */

              <img
                src={`${FACE_ENGINE_URL}/video_feed`}
                alt="AI Live CCTV Feed"
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  display: "block",
                }}
              />

            ) : (

              <div className="la-video__placeholder">

                <Video size={40} />

                <p>
                  Live CCTV Video Feed
                </p>

                <span>
                  Press "Start Attendance" to begin.
                </span>

              </div>

            )}


            {/* Camera label */}

            <div className="la-video__tag">

              <Camera size={14} />

              {sessionStatus === "live"
                ? "AI CCTV Feed"
                : "CCTV Feed"}

            </div>


            {/* Connection status */}

            <div className="la-video__connected">

              <span className="la-video__connected-dot" />

              {sessionStatus === "live"
                ? "AI Camera Connected"
                : "Camera Standby"}

            </div>

          </div>


          {/* =================================================
              START / STOP BUTTON
          ================================================= */}

          <div className="la-video__actions">

            {sessionStatus !== "live" ? (

              <button
                type="button"
                className="la-btn la-btn--green"
                onClick={handleStart}
              >

                <PlayCircle size={18} />

                Start Attendance

              </button>

            ) : (

              <button
                type="button"
                className="la-btn la-btn--red"
                onClick={handleStop}
              >

                <StopCircle size={18} />

                Stop Attendance

              </button>

            )}

          </div>


          {/* =================================================
              COUNTERS
          ================================================= */}

          <div className="la-counts">


            {/* Detected */}

            <div className="la-count">

              <span className="la-count__icon la-count__icon--blue">

                <Users size={18} />

              </span>

              <div>

                <strong>
                  {displayedDetectedFaces}
                </strong>

                <p>
                  Detected Faces
                </p>

              </div>

            </div>


            {/* Recognized */}

            <div className="la-count">

              <span className="la-count__icon la-count__icon--green">

                <UserCheck size={18} />

              </span>

              <div>

                <strong>
                  {recognized}
                </strong>

                <p>
                  Recognized Students
                </p>

              </div>

            </div>


            {/* Unknown */}

            <div className="la-count">

              <span className="la-count__icon la-count__icon--red">

                <UserX size={18} />

              </span>

              <div>

                <strong>
                  {unknown}
                </strong>

                <p>
                  Unknown Faces
                </p>

              </div>

            </div>

          </div>

        </div>


        {/* =================================================
            PIPELINE EXPLANATION
        ================================================= */}

        <div className="la-card la-pipeline-card">

          <h2>
            Recognition Pipeline
          </h2>


          <div className="la-pipeline">

            {PIPELINE_STEPS.map(
              (step, index) => (

                <React.Fragment
                  key={step}
                >

                  <div className="la-pipeline__step">
                    {step}
                  </div>


                  {index <
                    PIPELINE_STEPS.length - 1 && (

                    <ArrowRight
                      size={16}
                      className="la-pipeline__arrow"
                    />

                  )}

                </React.Fragment>

              )
            )}

          </div>


          <p className="la-pipeline__note">

            The live feed is processed by the
            Python face engine using YOLO,
            DeepSORT and FaceNet. Recognized
            students are sent to the Node.js
            attendance API and stored in MongoDB.

          </p>

        </div>

      </div>


      {/* =================================================
          ATTENDANCE LIST
      ================================================= */}

      <div className="la-card la-list-card">

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "12px",
            marginBottom: "16px",
          }}
        >

          <div>

            <h2>
              Today's Attendance
            </h2>

            <p
              style={{
                margin: "4px 0 0",
                color: "#667085",
                fontSize: "14px",
              }}
            >
              Students marked present by the AI system.
            </p>

          </div>


          <button
            type="button"
            className="la-btn la-btn--outline"
            onClick={handleRefresh}
            disabled={loadingAttendance}
          >

            <RefreshCw
              size={15}
              className={
                loadingAttendance
                  ? "la-spin"
                  : ""
              }
            />

            Refresh

          </button>

        </div>


        {/* =================================================
            TABLE
        ================================================= */}

        <table className="la-table">

          <thead>

            <tr>

              <th>
                Student ID
              </th>

              <th>
                Student Name
              </th>

              <th>
                Status
              </th>

              <th>
                Confidence
              </th>

              <th>
                Time
              </th>

            </tr>

          </thead>


          <tbody>

            {attendance.map(
              (row, index) => (

                <tr
                  key={
                    row._id ||
                    row.studentId ||
                    index
                  }
                >

                  <td>
                    {row.studentId || "--"}
                  </td>

                  <td>
                    {row.name || "--"}
                  </td>

                  <td>

                    <span
                      className="la-badge la-badge--present"
                    >
                      {row.status || "Present"}
                    </span>

                  </td>

                  <td>

                    {typeof row.similarity === "number"
                      ? `${(
                          row.similarity * 100
                        ).toFixed(1)}%`
                      : "--"}

                  </td>

                  <td>
                    {formatTime(row.time)}
                  </td>

                </tr>

              )
            )}

          </tbody>

        </table>


        {/* =================================================
            EMPTY STATE
        ================================================= */}

        {attendance.length === 0 && (

          <p className="la-empty">

            {loadingAttendance
              ? "Loading today's attendance..."
              : sessionStatus === "live"
              ? "No students have been marked present yet."
              : 'Press "Start Attendance" to begin the live recognition session.'}

          </p>

        )}


        {/* =================================================
            MOBILE CARDS
        ================================================= */}

        <div className="la-table-cards">

          {attendance.map(
            (row, index) => (

              <div
                className="la-table-card"
                key={
                  row._id ||
                  row.studentId ||
                  index
                }
              >

                <div className="la-table-card__head">

                  <strong>
                    {row.name || "--"}
                  </strong>

                  <span className="la-badge la-badge--present">
                    {row.status || "Present"}
                  </span>

                </div>


                <p className="la-table-card__meta">

                  {row.studentId || "--"}

                  {" • "}

                  {formatTime(row.time)}

                </p>


                <p className="la-table-card__meta">

                  Confidence:{" "}

                  {typeof row.similarity === "number"
                    ? `${(
                        row.similarity * 100
                      ).toFixed(1)}%`
                    : "--"}

                </p>

              </div>

            )
          )}

        </div>

      </div>


      {/* =================================================
          SMALL INFORMATION
      ================================================= */}

      <p
        style={{
          marginTop: "16px",
          color: "#667085",
          fontSize: "13px",
        }}
      >

        Face recognition runs on the Python
        face engine. Attendance records are
        retrieved from the Node.js backend.

      </p>

    </div>
  );
}