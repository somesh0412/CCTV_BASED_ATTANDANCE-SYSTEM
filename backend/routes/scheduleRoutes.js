const express = require("express");

const {
  getTodayTeacherSchedules,
  createTeacherSchedule,
} = require("../controllers/scheduleController");

const authMiddleware = require("../middleware/authMiddleware");
const roleMiddleware = require("../middleware/roleMiddleware");

const router = express.Router();

// ---------------------------------------------------------------------
// Teacher Today's Schedule
// GET /api/schedules/teacher/today
// ---------------------------------------------------------------------

router.get(
  "/teacher/today",
  authMiddleware,
  roleMiddleware("teacher"),
  getTodayTeacherSchedules
);

// ---------------------------------------------------------------------
// Teacher Create Schedule
// POST /api/schedules/teacher
// ---------------------------------------------------------------------

router.post(
  "/teacher",
  authMiddleware,
  roleMiddleware("teacher"),
  createTeacherSchedule
);

module.exports = router;