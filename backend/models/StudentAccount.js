/**
 * StudentAccount.js
 * -----------------------------------------------------------------------
 * Authentication account linked to an existing Student face record.
 * Keeping credentials separate prevents account registration from changing
 * face-recognition fields or creating a second student identity.
 * -----------------------------------------------------------------------
 */

const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");

const studentAccountSchema = new mongoose.Schema(
  {
    student: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Student",
      required: true,
      unique: true,
    },
    studentId: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      uppercase: true,
    },
    firstName: {
      type: String,
      required: true,
      trim: true,
      minlength: 2,
      maxlength: 50,
    },
    lastName: {
      type: String,
      required: true,
      trim: true,
      minlength: 1,
      maxlength: 50,
    },
    email: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      lowercase: true,
      match: [/^\S+@\S+\.\S+$/, "Please provide a valid email address"],
    },
    password: {
      type: String,
      required: true,
      minlength: 8,
      select: false,
    },
    department: {
      type: String,
      required: true,
      trim: true,
    },
    classDivision: {
      type: String,
      required: true,
      trim: true,
    },
    role: {
      type: String,
      enum: ["student"],
      default: "student",
    },
  },
  { timestamps: true }
);

studentAccountSchema.pre("save", async function hashPassword() {
  if (!this.isModified("password")) {
    return;
  }

  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
});

studentAccountSchema.methods.matchPassword = function matchPassword(enteredPassword) {
  return bcrypt.compare(enteredPassword, this.password);
};

studentAccountSchema.set("toJSON", {
  transform: (_doc, ret) => {
    delete ret.password;
    delete ret.__v;
    return ret;
  },
});

module.exports = mongoose.model("StudentAccount", studentAccountSchema);
