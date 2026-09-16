/**
 * authService.js
 * -----------------------------------------------------------------------
 * Contains the actual authentication business logic. Controllers stay
 * thin (HTTP-only concerns); this file talks to the Teacher model,
 * enforces business rules (duplicate checks, credential checks), and
 * returns plain data/throws errors for the controller to handle.
 * -----------------------------------------------------------------------
 */

const Teacher = require("../models/Teacher");
const generateToken = require("../utils/generateToken");

/**
 * Registers a new teacher.
 * Throws an error with .statusCode set for the error middleware to use.
 */
async function registerTeacher({ firstName, lastName, teacherId, department, email, password }) {
  const normalizedTeacherId = teacherId.trim().toUpperCase();
  const normalizedEmail = email.trim().toLowerCase();

  const existingByTeacherId = await Teacher.findOne({ teacherId: normalizedTeacherId });
  if (existingByTeacherId) {
    const error = new Error("Teacher ID already exists");
    error.statusCode = 409;
    throw error;
  }

  const existingByEmail = await Teacher.findOne({ email: normalizedEmail });
  if (existingByEmail) {
    const error = new Error("Email already exists");
    error.statusCode = 409;
    throw error;
  }

  // Password hashing happens automatically in the pre('save') hook on the model
  const teacher = await Teacher.create({
    firstName,
    lastName,
    teacherId: normalizedTeacherId,
    department,
    email: normalizedEmail,
    password,
  });

  return {
    id: teacher._id,
    firstName: teacher.firstName,
    lastName: teacher.lastName,
    teacherId: teacher.teacherId,
    department: teacher.department,
    email: teacher.email,
    role: teacher.role,
  };
}

/**
 * Authenticates a teacher and returns a JWT + basic profile info.
 */
async function loginTeacher({ teacherId, password }) {
  const normalizedTeacherId = teacherId.trim().toUpperCase();

  // password has `select: false` in the schema, so we explicitly request it
  const teacher = await Teacher.findOne({ teacherId: normalizedTeacherId }).select("+password");

  if (!teacher) {
    const error = new Error("Invalid teacher ID or password");
    error.statusCode = 401;
    throw error;
  }

  const isMatch = await teacher.matchPassword(password);
  if (!isMatch) {
    const error = new Error("Invalid teacher ID or password");
    error.statusCode = 401;
    throw error;
  }

  const token = generateToken(teacher);

  return {
    token,
    teacher: {
      id: teacher._id,
      firstName: teacher.firstName,
      lastName: teacher.lastName,
      teacherId: teacher.teacherId,
      department: teacher.department,
      email: teacher.email,
      role: teacher.role,
    },
  };
}

/**
 * Fetches a teacher's profile by id (used by the protected /profile route).
 */
async function getTeacherProfile(teacherIdFromToken) {
  const teacher = await Teacher.findById(teacherIdFromToken);

  if (!teacher) {
    const error = new Error("Teacher not found");
    error.statusCode = 401;
    throw error;
  }

  return {
    id: teacher._id,
    firstName: teacher.firstName,
    lastName: teacher.lastName,
    teacherId: teacher.teacherId,
    department: teacher.department,
    email: teacher.email,
    role: teacher.role,
  };
}

module.exports = { registerTeacher, loginTeacher, getTeacherProfile };
