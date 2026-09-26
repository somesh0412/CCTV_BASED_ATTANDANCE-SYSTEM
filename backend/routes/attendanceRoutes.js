/**
 * attendanceRoutes.js
 * -----------------------------------------------------------------------
 * Routes used by the Python face-recognition engine.
 * -----------------------------------------------------------------------
 */

const express = require("express");
const { body } = require("express-validator");

const attendanceController = require("../controllers/attendanceController");
const validateRequest = require("../middleware/validateMiddleware");
const authMiddleware = require("../middleware/authMiddleware");
const roleMiddleware = require("../middleware/roleMiddleware");

const router = express.Router();

// ---------------------------------------------------------------------
// POST /api/attendance
// ---------------------------------------------------------------------

const attendanceValidation = [
  body("studentId")
    .trim()
    .notEmpty()
    .withMessage("Student ID is required"),

  body("name")
    .trim()
    .notEmpty()
    .withMessage("Student name is required"),

  body("date")
    .trim()
    .notEmpty()
    .withMessage("Attendance date is required"),

  body("time")
    .trim()
    .notEmpty()
    .withMessage("Attendance time is required"),

  body("trackId")
    .isInt({ min: 0 })
    .withMessage("Track ID must be a valid number"),

  body("similarity")
    .isFloat({ min: 0, max: 1 })
    .withMessage("Similarity must be between 0 and 1"),
];

router.post(
  "/",
  attendanceValidation,
  validateRequest,
  attendanceController.createAttendance
);

// ---------------------------------------------------------------------
// GET /api/attendance
// ---------------------------------------------------------------------

router.get(
  "/",
  attendanceController.getAttendance
);

router.get(
  "/mine",
  authMiddleware,
  roleMiddleware("student"),
  attendanceController.getStudentAttendance
);

module.exports = router;