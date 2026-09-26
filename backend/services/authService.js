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
const Student = require("../models/Student");
const generateToken = require("../utils/generateToken");

function publicStudent(student) {
  return {
    id: student._id,
    firstName: student.firstName,
    lastName: student.lastName,
    name: student.name || `${student.firstName} ${student.lastName}`.trim(),
    studentId: student.studentId,
    department: student.department,
    email: student.email,
    faceRegistered: student.faceRegistered,
    role: student.role,
  };
}

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

async function registerStudent({ firstName, lastName, studentId, department, email, password, faceEmbeddings }) {
  const normalizedStudentId = studentId.trim().toUpperCase();
  const normalizedEmail = email.trim().toLowerCase();

  if (
    !Array.isArray(faceEmbeddings) ||
    faceEmbeddings.length === 0 ||
    faceEmbeddings.length > 10 ||
    faceEmbeddings.some(
      (embedding) =>
        !Array.isArray(embedding) ||
        embedding.length !== 512 ||
        embedding.some((value) => typeof value !== "number" || !Number.isFinite(value))
    )
  ) {
    const error = new Error("At least one valid 512-dimensional face embedding is required");
    error.statusCode = 400;
    throw error;
  }

  if (await Student.findOne({ studentId: normalizedStudentId })) {
    const error = new Error("Student ID already exists");
    error.statusCode = 409;
    throw error;
  }

  if (await Student.findOne({ email: normalizedEmail })) {
    const error = new Error("Email already exists");
    error.statusCode = 409;
    throw error;
  }

  const student = await Student.create({
    firstName,
    lastName,
    name: `${firstName.trim()} ${lastName.trim()}`,
    studentId: normalizedStudentId,
    department,
    email: normalizedEmail,
    password,
    faceRegistered: true,
    faceEmbeddings,
  });

  return publicStudent(student);
}

async function loginStudent({ studentId, password }) {
  const normalizedStudentId = studentId.trim().toUpperCase();
  const student = await Student.findOne({ studentId: normalizedStudentId }).select("+password");

  if (!student || !(await student.matchPassword(password))) {
    const error = new Error("Invalid student ID or password");
    error.statusCode = 401;
    throw error;
  }

  return { token: generateToken(student), student: publicStudent(student) };
}

async function getStudentProfile(studentIdFromToken) {
  const student = await Student.findById(studentIdFromToken);
  if (!student) {
    const error = new Error("Student not found");
    error.statusCode = 401;
    throw error;
  }
  return publicStudent(student);
}

module.exports = {
  registerTeacher,
  loginTeacher,
  getTeacherProfile,
  registerStudent,
  loginStudent,
  getStudentProfile,
};
