const mongoose = require("mongoose");

const studentEnrollmentSessionSchema = new mongoose.Schema(
  {
    student: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Student",
      required: true,
    },
    flaskSessionId: {
      type: String,
      required: true,
      unique: true,
      trim: true,
    },
    status: {
      type: String,
      enum: ["active", "completed", "cancelled", "failed"],
      default: "active",
    },
    embeddingsStored: {
      type: Boolean,
      default: false,
    },
    lastProgress: {
      type: mongoose.Schema.Types.Mixed,
      default: null,
    },
    completedAt: {
      type: Date,
      default: null,
    },
    expiresAt: {
      type: Date,
      default: () => new Date(Date.now() + 24 * 60 * 60 * 1000),
    },
  },
  { timestamps: true }
);

studentEnrollmentSessionSchema.index(
  { student: 1 },
  {
    unique: true,
    partialFilterExpression: { status: "active" },
  }
);

studentEnrollmentSessionSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

module.exports = mongoose.model(
  "StudentEnrollmentSession",
  studentEnrollmentSessionSchema
);
