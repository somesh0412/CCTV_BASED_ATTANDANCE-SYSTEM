/**
 * app.js
 * -----------------------------------------------------------------------
 * Express application configuration: middleware, routes, error handlers.
 * Does NOT connect to the DB or start listening — that's server.js's job.
 * Keeping this separate makes the app easy to import into tests later
 * (supertest etc.) without spinning up a real server/DB connection.
 * -----------------------------------------------------------------------
 */

const express = require("express");
const cors = require("cors");

const authRoutes = require("./routes/authRoutes");
const attendanceRoutes = require("./routes/attendanceRoutes");
const scheduleRoutes = require("./routes/scheduleRoutes");
const { notFound, errorHandler } = require("./middleware/errorMiddleware");

const app = express();

// --- Core middleware ---------------------------------------------------
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// --- Health check --------------------------------------------------------
app.get("/api/health", (req, res) => {
  res.status(200).json({ success: true, message: "API is running" });
});

// --- Routes --------------------------------------------------------------
app.use("/api/auth", authRoutes);
app.use("/api/attendance", attendanceRoutes);
app.use("/api/schedules", scheduleRoutes);

// --- Error handling (must be last) ---------------------------------------
app.use(notFound);
app.use(errorHandler);

module.exports = app;
