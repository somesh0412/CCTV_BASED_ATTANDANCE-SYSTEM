# CCTV-Based Attendance System

A full-stack attendance platform that combines a React web application, an Express/MongoDB API, and a Python computer-vision engine. The system supports teacher and student accounts, student face enrollment, scheduled attendance sessions, live face recognition, attendance reports, and CSV export.

## Project Status

This repository is an active development project. The web application and Node.js API are organized as separate packages. The face engine contains reusable detection, tracking, recognition, enrollment, and attendance modules, plus camera and diagnostic scripts.

Important current behavior:

- Attendance events are accepted by `POST /api/attendance` only while a teacher attendance session is live.
- The Python attendance engine prevents duplicates in memory during its process, while the Node.js API also prevents duplicates per persisted attendance session.
- Student face enrollment is brokered by the Node.js API and requires the Flask enrollment service to be running.
- No Python dependency manifest is currently committed. Create and maintain a Python virtual environment manually until one is added.

## Architecture

```text
React/Vite frontend
        |
        | Axios + JWT
        v
Express API (backend)
        |
        +--> MongoDB via Mongoose
        |
        +--> Flask face-enrollment service
        |
        ^
        | attendance events
Python face engine
  webcam/RTSP -> YOLO face detection -> DeepSORT tracking
              -> face crop -> FaceNet embedding -> matcher
              -> identity cache -> attendance engine -> Node API
```

### Components

- `frontend/`: React 19 and Vite client. Provides landing, teacher authentication/dashboard, student authentication/dashboard, face registration, live attendance, history, reports, schedules, and profile views.
- `backend/`: Express 5 API. Owns authentication, authorization, validation, attendance sessions and records, schedules, student enrollment sessions, and MongoDB persistence.
- `face-engine/`: Python computer-vision code. Contains webcam/stream handling, YOLO detection, DeepSORT tracking, FaceNet embeddings, identity matching, enrollment, attendance decisions, and API clients.

## Repository Layout

```text
.
├── backend/
│   ├── app.js                         Express middleware and route registration
│   ├── server.js                      MongoDB connection and HTTP startup
│   ├── config/                        Environment and database configuration
│   ├── controllers/                   Request handlers and business workflows
│   ├── middleware/                    Auth, role, validation, and error handling
│   ├── models/                        Mongoose models
│   ├── routes/                        API route definitions
│   ├── services/                      Authentication services
│   └── utils/                         Tokens, responses, class and schedule helpers
├── face-engine/
│   ├── attendance/                    In-memory attendance decision engine
│   ├── backend/                       Python client for the Node attendance API
│   ├── camera/                        Webcam and stream access
│   ├── database/                      MongoDB/student repository helpers
│   ├── detection/                     YOLO face detection
│   ├── enrollment/                    Face enrollment workflow
│   ├── face_processing/               Face crop and quality checks
│   ├── identity/                      Track identity cache
│   ├── pipeline/                      Face and attendance pipelines
│   ├── recognition/                   Embeddings and matching
│   ├── tracking/                      DeepSORT tracking
│   └── testing/                       Evaluation and ground-truth tools
├── frontend/
│   ├── src/api/                       Shared Axios client
│   ├── src/components/                Reusable route/components
│   ├── src/pages/                     Teacher and student screens
│   └── public/                        Static public assets
└── AGENT.md                           Development handoff and change log
```

## Prerequisites

Install the following before running the complete system:

- Node.js and npm
- MongoDB, local or hosted
- Python 3 with a virtual environment
- A working webcam or compatible CCTV/RTSP stream for recognition
- The Python packages required by the imported modules, including OpenCV, NumPy, PyTorch/torchvision, `facenet-pytorch`, Ultralytics, DeepSORT, Flask, Flask-CORS, PyMongo, `requests`, `python-dotenv`, Pillow, and ONVIF support as needed by the selected scripts

The repository currently does not contain `requirements.txt` or `pyproject.toml`; record the chosen Python package versions when setting up a development machine.

## Configuration

### Backend environment

Create `backend/.env` with the required values:

```env
PORT=5000
MONGO_URI=mongodb://127.0.0.1:27017/cctv_attendance
JWT_SECRET=replace-with-a-long-random-secret
JWT_EXPIRES_IN=1d
NODE_ENV=development
FACE_ENROLLMENT_API_URL=http://127.0.0.1:8000
FACE_ENROLLMENT_API_SECRET=replace-with-the-shared-enrollment-secret
```

`PORT`, `MONGO_URI`, `JWT_SECRET`, and `JWT_EXPIRES_IN` are required by `backend/config/env.js`. The face-enrollment URL defaults to port `8000`, but enrollment also requires `FACE_ENROLLMENT_API_SECRET` to be configured.

### Frontend environment

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:5000/api
```

The shared Axios client reads `VITE_API_URL` and adds the JWT stored as `token` in browser local storage.

### Face engine configuration

The integrated attendance pipeline currently sends events to:

```text
http://localhost:5000/api/attendance
```

The face enrollment Flask service is called by the Node API at `FACE_ENROLLMENT_API_URL`. Keep the enrollment secret identical between the Node API and Flask service. Do not commit credentials, biometric datasets, embeddings, or model files that are excluded by `.gitignore`.

## Installation and Running

Run each service from its own directory.

### 1. Install frontend dependencies

```powershell
cd frontend
npm install
npm run dev
```

Vite prints the local frontend URL, normally `http://localhost:5173`.

### 2. Install backend dependencies and start the API

```powershell
cd backend
npm install
npm run dev
```

The backend starts after it connects to MongoDB. The health endpoint is:

```text
GET http://localhost:5000/api/health
```

The backend `start` script currently invokes `server.js` directly; use `npm run dev` during development or run `node server.js` explicitly for a production-style start.

### 3. Start the Python face-enrollment service

The Flask service implementation is in `face-engine/backend/` and is used by student face registration. Its exact launch command should be confirmed from the selected Flask entry script before deployment. Ensure it listens on the URL configured in `FACE_ENROLLMENT_API_URL` and exposes the enrollment endpoints listed below.

### 4. Run the integrated recognition pipeline

From `face-engine/`, activate the Python virtual environment and run the relevant pipeline or camera script. The main integration is `pipeline/attendance_pipeline.py`; it loads the face pipeline, marks each recognized student once per process, and posts events to the Node API.

The pipeline requires:

- YOLO model assets, including the repository's face model path
- Student embeddings available to the student repository
- A reachable camera/webcam or stream
- A live backend attendance session before events can be persisted

## Backend API Surface

All routes are prefixed with `/api`.

### Health and authentication

- `GET /health`
- `POST /auth/teacher/register`
- `POST /auth/teacher/login`
- `GET /auth/teacher/profile` (teacher JWT)
- `PUT /auth/teacher/profile` (teacher JWT)
- `POST /auth/student/register`
- `POST /auth/student/login`
- `GET /auth/student/dashboard` (student JWT)

### Attendance

- `POST /attendance` receives a face-engine event. It requires an active attendance session and validates the recognized student and class eligibility.
- `GET /attendance` (teacher JWT)
- `GET /attendance/teacher/history` (teacher JWT)
- `POST /attendance/sessions/start` (teacher JWT)
- `GET /attendance/teacher/reports` (teacher JWT)
- `GET /attendance/teacher/reports/export` (teacher JWT)

### Schedules

- `GET /schedules/teacher/today` (teacher JWT)
- `POST /schedules/teacher` (teacher JWT)

### Student face enrollment

- `POST /student-enrollment/start` (student JWT)
- `POST /student-enrollment/process-frame` (student JWT, multipart field `frame`)
- `GET /student-enrollment/progress/:sessionId` (student JWT)
- `POST /student-enrollment/cancel/:sessionId` (student JWT)

## Recognition and Attendance Flow

1. The camera module obtains a frame.
2. YOLO detects faces.
3. DeepSORT assigns tracking IDs.
4. A face crop is produced for each tracked face.
5. FaceNet creates an embedding for a new track.
6. The matcher compares the embedding with loaded student embeddings.
7. The identity cache reuses a recent track identity to avoid repeated matching.
8. The attendance engine ignores invalid or `UNKNOWN` identities and suppresses duplicates in its process.
9. The Python API client posts the event to the Node API.
10. The Node API checks the live session, schedule, student record, and class eligibility before writing MongoDB attendance.

## Face Enrollment Flow

1. An authenticated student starts an enrollment session in the frontend.
2. The Node API creates a persisted enrollment session and calls Flask `/enrollment/start`.
3. The frontend uploads camera frames as multipart data to the Node API.
4. The Node API forwards each frame to Flask `/enrollment/process-frame`.
5. When Flask returns a completed session, the Node API validates at least 25 embeddings, each with 512 finite numeric values.
6. The embeddings are saved to the student record and the enrollment session is marked completed.

## Validation and Tests

Frontend checks:

```powershell
cd frontend
npm run lint
npm run build
```

Backend smoke check:

```powershell
Invoke-WebRequest http://localhost:5000/api/health
```

Python checks are script-based rather than managed by a single test runner. Relevant files include `face-engine/enrollment/test_enrollment_policy.py`, `face-engine/test_mongodb.py`, camera/stream tests, and tools under `face-engine/testing/`. Hardware, model files, MongoDB, and service availability may be required.

## Security and Data Notes

- Keep `.env` files and secrets out of version control.
- Treat face embeddings and captured frames as biometric data.
- Use HTTPS and secure secret management outside local development.
- Restrict CORS before production deployment; the current backend enables broad CORS.
- The Python attendance endpoint is intentionally callable without JWT because the face engine posts directly to it. Protect the deployment network and add service authentication before exposing it beyond a trusted environment.
- Never commit student datasets or generated embeddings.

## Known Gaps

- Python dependencies are not pinned in a manifest.
- The root package does not provide commands to start all services together.
- Flask service startup and its complete endpoint contract are not documented in a dedicated Python README.
- Face-engine paths and camera settings are partly script-specific and may require local configuration.
- Automated API and end-to-end tests are limited; most face-engine checks depend on hardware or external services.

## Working With This Repository

Read `AGENT.md` before changing code. Keep frontend, backend, and face-engine changes separated where possible, update this README when setup or contracts change, and add a change-log entry to `AGENT.md` for meaningful work.
