/**
 * scheduleController.js
 * -----------------------------------------------------------------------
 * Schedule operations for the Teacher Dashboard.
 * -----------------------------------------------------------------------
 */

const Schedule = require("../models/Schedule");
const Teacher = require("../models/Teacher");
const { parseClassDivision } = require("../utils/classIdentity");
const {
  formatTime12,
  getIndiaDateString,
  normalizeScheduleTime,
} = require("../utils/scheduleTime");

// ---------------------------------------------------------------------
// GET /api/schedules/teacher/today
// Get today's schedules for logged-in teacher
// ---------------------------------------------------------------------

const getTodayTeacherSchedules = async (req, res, next) => {
  try {
    const teacher = await Teacher.findById(req.user.id).select("teacherId");

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    const today = getIndiaDateString();

    const schedules = await Schedule.find({
      teacher: teacher.teacherId,
      date: today,
    }).sort({ createdAt: 1 });

    const formattedSchedules = schedules.map((schedule) => {
      let startTime = "";
      let endTime = "";

      // Stored format:
      // "10:15 AM - 11:15 AM"
      if (schedule.time && schedule.time.includes(" - ")) {
        const parts = schedule.time.split(" - ");

        startTime = formatTime12(parts[0]);
        endTime = formatTime12(parts[1]);
      }

      return {
        id: schedule._id,
        startTime,
        endTime,
        subject: schedule.subject,
        classDivision: schedule.classDivision,
        room: schedule.room,
        status: schedule.status,
        date: schedule.date,
      };
    });

    return res.status(200).json({
      success: true,
      count: formattedSchedules.length,
      schedules: formattedSchedules,
    });
  } catch (error) {
    next(error);
  }
};

// ---------------------------------------------------------------------
// POST /api/schedules/teacher
// Create schedule for logged-in teacher
// ---------------------------------------------------------------------

const createTeacherSchedule = async (req, res, next) => {
  try {
    const teacher = await Teacher.findById(req.user.id).select(
      "teacherId department"
    );

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    const {
      subject,
      classDivision,
      startTime,
      endTime,
      room,
      date,
      department,
      academicYear,
      division,
    } = req.body;

    // ---------------------------------------------------------------
    // Validation
    // ---------------------------------------------------------------

    if (
      !subject ||
      !classDivision ||
      !startTime ||
      !endTime ||
      !room
    ) {
      return res.status(400).json({
        success: false,
        message:
          "subject, classDivision, startTime, endTime and room are required",
      });
    }

    // ---------------------------------------------------------------
    // Combine start and end time
    // ---------------------------------------------------------------

    const time = normalizeScheduleTime(startTime, endTime);
    if (!time) {
      return res.status(400).json({
        success: false,
        message: "Start time and end time must be valid 12-hour or 24-hour times.",
      });
    }

    const [normalizedStart, normalizedEnd] = time.split(" - ");
    if (normalizedStart === normalizedEnd) {
      return res.status(400).json({
        success: false,
        message: "End time must be later than start time.",
      });
    }
    const parsedClass = parseClassDivision(classDivision);

    const schedule = await Schedule.create({
      subject,
      time,
      room,
      teacher: teacher.teacherId,
      classDivision,
      department: department || undefined,
      academicYear: academicYear || parsedClass.year || undefined,
      division: division || parsedClass.division || undefined,
      date: date || getIndiaDateString(),
      status: "upcoming",
    });

    // Return same shape expected by frontend
    const responseSchedule = {
      id: schedule._id,
      startTime: formatTime12(normalizedStart),
      endTime: formatTime12(normalizedEnd),
      subject: schedule.subject,
      classDivision: schedule.classDivision,
      department: schedule.department,
      academicYear: schedule.academicYear,
      division: schedule.division,
      room: schedule.room,
      status: schedule.status,
      date: schedule.date,
    };

    return res.status(201).json({
      success: true,
      message: "Schedule created successfully",
      schedule: responseSchedule,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  getTodayTeacherSchedules,
  createTeacherSchedule,
};