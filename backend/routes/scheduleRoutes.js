/**
 * scheduleRoutes.js
 * -----------------------------------------------------------------------
 * Express CRUD routes for schedules (/api/schedules).
 * -----------------------------------------------------------------------
 */

const express = require("express");
const { body } = require("express-validator");
const scheduleController = require("../controllers/scheduleController");
const validateRequest = require("../middleware/validateMiddleware");

const router = express.Router();

// Validation for creating/updating a schedule
const scheduleValidation = [
  body("subject")
    .trim()
    .notEmpty()
    .withMessage("Subject is required"),

  body("time")
    .trim()
    .notEmpty()
    .withMessage("Time slot is required"),

  body("room")
    .trim()
    .notEmpty()
    .withMessage("Room is required"),

  body("teacher")
    .trim()
    .notEmpty()
    .withMessage("Teacher is required"),
];

// CRUD Routes
router.post(
  "/",
  scheduleValidation,
  validateRequest,
  scheduleController.createSchedule
);

router.get("/", scheduleController.getSchedules);

router.get("/:id", scheduleController.getScheduleById);

router.put("/:id", scheduleController.updateSchedule);

router.delete("/:id", scheduleController.deleteSchedule);

module.exports = router;
