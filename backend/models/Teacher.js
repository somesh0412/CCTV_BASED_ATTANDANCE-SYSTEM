/**
 * Teacher.js
 * -----------------------------------------------------------------------
 * Mongoose model for a Teacher. Handles:
 *  - schema/validation
 *  - unique teacherId + email
 *  - password hashing via bcrypt (pre-save hook)
 *  - a matchPassword() instance method for login
 *  - stripping password out of toJSON() output automatically
 * -----------------------------------------------------------------------
 */

const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");

const teacherSchema = new mongoose.Schema(
  {
    firstName: {
      type: String,
      required: [true, "First name is required"],
      trim: true,
      minlength: [2, "First name must be at least 2 characters"],
      maxlength: [50, "First name cannot exceed 50 characters"],
    },
    lastName: {
      type: String,
      required: [true, "Last name is required"],
      trim: true,
      minlength: [1, "Last name must be at least 1 character"],
      maxlength: [50, "Last name cannot exceed 50 characters"],
    },
    teacherId: {
      type: String,
      required: [true, "Teacher ID is required"],
      unique: true,
      trim: true,
      uppercase: true,
    },
    department: {
      type: String,
      required: [true, "Department is required"],
      trim: true,
    },
    email: {
      type: String,
      required: [true, "Email is required"],
      unique: true,
      trim: true,
      lowercase: true,
      match: [/^\S+@\S+\.\S+$/, "Please provide a valid email address"],
    },
    password: {
      type: String,
      required: [true, "Password is required"],
      minlength: [6, "Password must be at least 6 characters"],
      select: false, // never returned by default in queries
    },
    role: {
      type: String,
      enum: ["teacher"],
      default: "teacher",
    },
  },
  {
    timestamps: { createdAt: true, updatedAt: false },
  }
);

// Hash password before saving, only if it was modified
teacherSchema.pre("save", async function () {

  if (!this.isModified("password")) {
    return;
  }

  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);

});
// Instance method: compare plaintext password against stored hash
teacherSchema.methods.matchPassword = async function matchPassword(enteredPassword) {
  return bcrypt.compare(enteredPassword, this.password);
};

// Ensure password never leaks out, even if someone forgets .select('-password')
teacherSchema.set("toJSON", {
  transform: (_doc, ret) => {
    delete ret.password;
    delete ret.__v;
    return ret;
  },
});

module.exports = mongoose.model("Teacher", teacherSchema);
