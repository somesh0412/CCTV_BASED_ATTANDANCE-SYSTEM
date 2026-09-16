import axios from "axios";

/**
 * Centralized Axios instance.
 *
 * Requires a `.env` file in the frontend root:
 *
 *   VITE_API_URL=http://localhost:5000/api
 *
 * Based on authRoutes.js, the teacher endpoints are mounted at:
 *   POST /api/auth/teacher/register
 *   POST /api/auth/teacher/login
 *   GET  /api/auth/teacher/profile
 *
 * So calls in this project should use paths like:
 *   api.post("/auth/teacher/register", { ... })
 *   api.post("/auth/teacher/login", { ... })
 */
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Request interceptor
 * Attaches the stored JWT (if present) to every outgoing request so
 * protected teacher routes (e.g. GET /auth/teacher/profile) work
 * automatically without repeating this logic everywhere.
 */
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

/**
 * Response interceptor
 * If the backend returns 401 (invalid/expired JWT), clear local auth
 * state and send the teacher back to login. Guarded against redirect
 * loops by only redirecting when not already on the login page.
 */
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("teacher");

      if (window.location.pathname !== "/teacher-login") {
        window.location.href = "/teacher-login";
      }
    }
    return Promise.reject(error);
  }
);

export default api;