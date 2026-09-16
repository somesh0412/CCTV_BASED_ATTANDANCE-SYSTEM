/**
 * attendanceController.js
 * -----------------------------------------------------------------------
 * Handles attendance events received from the Python face engine.
 * -----------------------------------------------------------------------
 */

const Attendance = require("../models/Attendance");

// ---------------------------------------------------------------------
// POST /api/attendance
// ---------------------------------------------------------------------

const createAttendance = async (req, res, next) => {
  try {
    const {
      studentId,
      name,
      date,
      time,
      status,
      trackId,
      similarity,
    } = req.body;

    // ---------------------------------------------------------------
    // Basic validation
    // ---------------------------------------------------------------

    if (
      !studentId ||
      !name ||
      !date ||
      !time ||
      trackId === undefined ||
      similarity === undefined
    ) {
      return res.status(400).json({
        success: false,
        message:
          "studentId, name, date, time, trackId and similarity are required",
      });
    }

    // ---------------------------------------------------------------
    // Do not accept UNKNOWN
    // ---------------------------------------------------------------

    if (studentId.toUpperCase() === "UNKNOWN") {
      return res.status(400).json({
        success: false,
        message: "Unknown students cannot be marked present",
      });
    }

    // ---------------------------------------------------------------
    // Check whether already marked today
    // ---------------------------------------------------------------

    const existingAttendance = await Attendance.findOne({
      studentId,
      date,
    });

    if (existingAttendance) {
      return res.status(200).json({
        success: true,
        alreadyMarked: true,
        message: "Attendance already marked for this student",
        attendance: existingAttendance,
      });
    }

    // ---------------------------------------------------------------
    // Create attendance
    // ---------------------------------------------------------------

    const attendance = await Attendance.create({
      studentId,
      name,
      date,
      time,
      status: status || "Present",
      trackId,
      similarity,
    });

    return res.status(201).json({
      success: true,
      alreadyMarked: false,
      message: "Attendance marked successfully",
      attendance,
    });
  } catch (error) {
    // ---------------------------------------------------------------
    // Handle MongoDB duplicate-key race condition
    // ---------------------------------------------------------------

    if (error.code === 11000) {
      return res.status(200).json({
        success: true,
        alreadyMarked: true,
        message: "Attendance already marked for this student today",
      });
    }

    next(error);
  }
};

// ---------------------------------------------------------------------
// GET /api/attendance
// ---------------------------------------------------------------------

const getAttendance = async (req, res, next) => {
  try {
    const { date } = req.query;
    const filter = date ? { date } : {};
    const attendance = await Attendance.find(filter)
      .sort({ date: -1, time: -1 });

    return res.status(200).json({
      success: true,
      count: attendance.length,
      attendance,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  createAttendance,
  getAttendance,
};