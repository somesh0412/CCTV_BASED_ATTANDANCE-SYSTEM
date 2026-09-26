/**
 * scheduleController.js
 * -----------------------------------------------------------------------
 * Controller for Schedule CRUD operations.
 * -----------------------------------------------------------------------
 */

const Schedule = require("../models/Schedule");

// ---------------------------------------------------------------------
// POST /api/schedules - Create a new schedule entry
// ---------------------------------------------------------------------
const createSchedule = async (req, res, next) => {
  try {
    const { subject, time, room, teacher, classDivision, date, status } = req.body;

    if (!subject || !time || !room || !teacher) {
      return res.status(400).json({
        success: false,
        message: "subject, time, room, and teacher are required fields",
      });
    }

    const newSchedule = await Schedule.create({
      subject,
      time,
      room,
      teacher,
      classDivision: classDivision || "All Divisions",
      date: date || new Date().toISOString().split("T")[0],
      status: status || "upcoming",
    });

    return res.status(201).json({
      success: true,
      message: "Schedule entry created successfully",
      schedule: newSchedule,
    });
  } catch (error) {
    next(error);
  }
};

// ---------------------------------------------------------------------
// GET /api/schedules - Get all schedules (with optional date & teacher filter)
// ---------------------------------------------------------------------
const getSchedules = async (req, res, next) => {
  try {
    const { date, teacher } = req.query;
    const filter = {};

    if (date) {
      filter.date = date;
    }
    if (teacher) {
      filter.teacher = new RegExp(teacher, "i");
    }

    const schedules = await Schedule.find(filter).sort({ createdAt: -1 });

    return res.status(200).json({
      success: true,
      count: schedules.length,
      schedules,
    });
  } catch (error) {
    next(error);
  }
};

// ---------------------------------------------------------------------
// GET /api/schedules/:id - Get a single schedule by ID
// ---------------------------------------------------------------------
const getScheduleById = async (req, res, next) => {
  try {
    const schedule = await Schedule.findById(req.params.id);

    if (!schedule) {
      return res.status(404).json({
        success: false,
        message: "Schedule not found",
      });
    }

    return res.status(200).json({
      success: true,
      schedule,
    });
  } catch (error) {
    next(error);
  }
};

// ---------------------------------------------------------------------
// PUT /api/schedules/:id - Update schedule by ID
// ---------------------------------------------------------------------
const updateSchedule = async (req, res, next) => {
  try {
    const schedule = await Schedule.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    );

    if (!schedule) {
      return res.status(404).json({
        success: false,
        message: "Schedule not found",
      });
    }

    return res.status(200).json({
      success: true,
      message: "Schedule updated successfully",
      schedule,
    });
  } catch (error) {
    next(error);
  }
};

// ---------------------------------------------------------------------
// DELETE /api/schedules/:id - Delete schedule by ID
// ---------------------------------------------------------------------
const deleteSchedule = async (req, res, next) => {
  try {
    const schedule = await Schedule.findByIdAndDelete(req.params.id);

    if (!schedule) {
      return res.status(404).json({
        success: false,
        message: "Schedule not found",
      });
    }

    return res.status(200).json({
      success: true,
      message: "Schedule deleted successfully",
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  createSchedule,
  getSchedules,
  getScheduleById,
  updateSchedule,
  deleteSchedule,
};
