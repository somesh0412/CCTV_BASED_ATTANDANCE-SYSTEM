/**
 * AttendanceSession.js
 * -----------------------------------------------------------------------
 * Represents one attendance session started by a teacher for a class.
 * -----------------------------------------------------------------------
 */

const mongoose = require("mongoose");

const attendanceSessionSchema = new mongoose.Schema(
  {
    // Teacher who started the session
    teacherId: {
      type: String,
      required: true,
      trim: true,
      uppercase: true,
    },

    // Schedule from which this attendance session was started
    scheduleId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Schedule",
      required: true,
    },

    subject: {
      type: String,
      required: true,
      trim: true,
    },

    classDivision: {
      type: String,
      required: true,
      trim: true,
    },

    department: {
      type: String,
      trim: true,
    },

    academicYear: {
      type: String,
      trim: true,
    },

    division: {
      type: String,
      trim: true,
    },

    room: {
      type: String,
      required: true,
      trim: true,
    },

    // YYYY-MM-DD
    date: {
      type: String,
      required: true,
      trim: true,
    },

    startTime: {
      type: String,
      required: true,
      trim: true,
    },

    endTime: {
      type: String,
      required: true,
      trim: true,
    },

    status: {
      type: String,
      enum: ["live", "completed", "stopped"],
      default: "live",
    },

    startedAt: {
      type: Date,
      default: Date.now,
    },

    endedAt: {
      type: Date,
      default: null,
    },
  },
  {
    timestamps: true,
  }
);

// Useful for teacher attendance history
attendanceSessionSchema.index({
  teacherId: 1,
  date: -1,
});

// Useful for finding sessions by schedule
attendanceSessionSchema.index({
  scheduleId: 1,
  date: -1,
});

module.exports = mongoose.model(
  "AttendanceSession",
  attendanceSessionSchema
);