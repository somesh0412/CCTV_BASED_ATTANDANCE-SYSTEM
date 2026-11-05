const express = require("express");

const {
  createAttendance,
  getAttendance,
  getTeacherAttendanceHistory,
  startAttendanceSession,
  getTeacherReports,
  exportTeacherReports,
} = require("../controllers/attendanceController");

const authMiddleware = require("../middleware/authMiddleware");
const roleMiddleware = require("../middleware/roleMiddleware");

const router = express.Router();

// Python Face Engine
// POST /api/attendance
router.post("/", createAttendance);

// General attendance
// GET /api/attendance
router.get(
  "/",
  authMiddleware,
  roleMiddleware("teacher"),
  getAttendance
);

// Teacher Attendance History
// GET /api/attendance/teacher/history
router.get(
  "/teacher/history",
  authMiddleware,
  roleMiddleware("teacher"),
  getTeacherAttendanceHistory
);
// Start attendance session
// POST /api/attendance/sessions/start

router.post(
  "/sessions/start",
  authMiddleware,
  roleMiddleware("teacher"),
  startAttendanceSession
);
// ---------------------------------------------------------------------
// Teacher Reports
// GET /api/attendance/teacher/reports
// ---------------------------------------------------------------------

router.get(
  "/teacher/reports",
  authMiddleware,
  roleMiddleware("teacher"),
  getTeacherReports
);
// ---------------------------------------------------------------------
// Export teacher attendance report
// GET /api/attendance/teacher/reports/export
// ---------------------------------------------------------------------

router.get(
  "/teacher/reports/export",
  authMiddleware,
  roleMiddleware("teacher"),
  exportTeacherReports
);
module.exports = router;