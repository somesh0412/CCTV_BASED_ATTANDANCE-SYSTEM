/**
 * Schedule.js
 * -----------------------------------------------------------------------
 * Mongoose model for class timetable schedules.
 * -----------------------------------------------------------------------
 */

const mongoose = require("mongoose");

const scheduleSchema = new mongoose.Schema(
  {
    subject: {
      type: String,
      required: [true, "Subject is required"],
      trim: true,
    },
    time: {
      type: String,
      required: [true, "Time slot is required"],
      trim: true,
    },
    room: {
      type: String,
      required: [true, "Room is required"],
      trim: true,
    },
    teacher: {
      type: String,
      required: [true, "Teacher name or ID is required"],
      trim: true,
    },
    classDivision: {
      type: String,
      trim: true,
      default: "All Divisions",
    },
    date: {
      type: String,
      trim: true,
      default: () => new Date().toISOString().split("T")[0],
    },
    status: {
      type: String,
      enum: ["upcoming", "live", "completed"],
      default: "upcoming",
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Schedule", scheduleSchema);
