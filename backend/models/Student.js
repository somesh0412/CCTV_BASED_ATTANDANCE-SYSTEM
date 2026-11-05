/**
 * Student.js
 * -----------------------------------------------------------------------
 * Mongoose shape for the face-recognition student master collection.
 * Authentication data is intentionally kept in StudentAccount.
 * -----------------------------------------------------------------------
 */

const mongoose = require("mongoose");

const studentSchema = new mongoose.Schema(
  {
    studentId: {
      type: String,
      required: true,
      trim: true,
      uppercase: true,
    },
    firstName: {
      type: String,
      trim: true,
      minlength: 2,
      maxlength: 50,
    },
    lastName: {
      type: String,
      trim: true,
      minlength: 1,
      maxlength: 50,
    },
    name: String,
    email: {
      type: String,
      trim: true,
      lowercase: true,
      match: /^\S+@\S+\.\S+$/,
    },
    department: {
      type: String,
      trim: true,
    },
    className: {
      type: String,
      trim: true,
    },
    faceRegistered: Boolean,
    faceEmbeddings: {
      type: [mongoose.Schema.Types.Mixed],
      default: [],
    },
  },
  {
    collection: "students",
    strict: false,
    timestamps: false,
  }
);

module.exports = mongoose.model("Student", studentSchema);
