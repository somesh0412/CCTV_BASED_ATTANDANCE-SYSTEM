# CCTV Based Attendance System

A CCTV and face-recognition attendance system with a Node.js/Express backend, React frontend, MongoDB storage, and Python face-recognition services.

## Project Areas

- `frontend/` - React and Vite user interface for teachers and students.
- `backend/` - Express API, authentication, attendance, schedules, and MongoDB models.
- `face-engine/` - Face detection, recognition, tracking, camera streaming, and attendance processing.

## Current Branch Work

The `s-project-development` branch is based on `project-development` and includes:

- Student registration, login, and dashboard pages.
- Student authentication routes and backend model support.
- Updated attendance and authentication services.
- Face-engine pipeline, camera-streaming, and API updates.
- Environment example files for backend and face-engine configuration.
- Runtime architecture documentation and visual-check artifacts.

For the complete file-level summary and merge checklist, see [S_PROJECT_DEVELOPMENT_CHANGES.md](S_PROJECT_DEVELOPMENT_CHANGES.md).

## Configuration

Create local `.env` files from the provided examples before running the services:

- `backend/.env.example`
- `face-engine/.env.example`

Keep passwords, tokens, database URLs, and other secrets out of Git.

The frontend uses `VITE_API_URL` for the backend API URL and defaults to `http://localhost:5000/api` during local development.

## Before Merging

Verify teacher login, student registration and login, attendance API requests, MongoDB connectivity, face-engine camera processing, frontend linting, and the production build before merging `s-project-development` into `project-development`.
