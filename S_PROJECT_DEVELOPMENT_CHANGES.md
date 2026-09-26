# s-project-development Changes

This branch is based on `project-development` and contains the additional work added in the local project snapshot.

## What Was Added or Updated

### Student Experience

- Added student login and registration pages.
- Added routes for `/student-login` and `/student-register`.
- Updated the landing page so the Student Login button opens the student login page.
- Updated the student dashboard styling and behavior.
- Added the `Student` backend model.

### Backend

- Added student-related authentication and route handling.
- Updated authentication services, token generation, and auth controllers.
- Updated attendance controllers and routes.
- Added `backend/.env.example` to document backend environment variables.
- Updated backend package dependencies and lockfile.

### Face Engine

- Updated attendance API integration.
- Updated the attendance pipeline and camera streaming flow.
- Added `face-engine/requirements.txt` for Python dependencies.
- Added `face-engine/.env.example` for environment configuration.

### Architecture Documentation

- Added the generated runtime architecture diagram and supporting JSON files.
- Added visual-check HTML, JSON, and image artifacts.

## Important Configuration

Copy the environment example files to local `.env` files and fill in the real values before running the application. Do not commit secrets.

The frontend API client uses `VITE_API_URL` when it is configured and otherwise defaults to `http://localhost:5000/api` for local development.

## Suggested Verification

Before merging this branch into `project-development`, verify:

1. Teacher login and dashboard behavior.
2. Student registration, login, and dashboard behavior.
3. Backend authentication and attendance API requests.
4. MongoDB connectivity and environment configuration.
5. Face-engine dependencies, camera streaming, and attendance recording.
6. Frontend lint and production build.
