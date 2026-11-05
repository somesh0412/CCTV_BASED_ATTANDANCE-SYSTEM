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
const StudentAccount = require("../models/StudentAccount");
const mongoose = require("mongoose");
const generateToken = require("../utils/generateToken");
const generateStudentToken = require("../utils/generateStudentToken");

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

async function registerStudent({
  firstName,
  lastName,
  studentId,
  email,
  password,
  department,
  classDivision,
}) {
  const normalizedStudentId = studentId.trim().toUpperCase();
  const normalizedEmail = email.trim().toLowerCase();

  const existingStudent = await Student.findOne({ studentId: normalizedStudentId }).select(
    "_id studentId firstName lastName name email department className faceRegistered"
  );

  if (!existingStudent) {
    const error = new Error("Student record not found. Please complete face registration first.");
    error.statusCode = 404;
    throw error;
  }

  const accountEmail = (existingStudent.email || normalizedEmail).trim().toLowerCase();
  const existingAccount = await StudentAccount.findOne({
    $or: [{ student: existingStudent._id }, { studentId: normalizedStudentId }, { email: accountEmail }],
  }).select("_id");

  if (existingAccount) {
    const error = new Error("Student account already exists. Please log in.");
    error.statusCode = 409;
    throw error;
  }

  const existingNameParts = (existingStudent.name || "").trim().split(/\s+/);
  const accountFirstName =
    existingStudent.firstName || existingNameParts[0] || firstName;
  const accountLastName =
    existingStudent.lastName || existingNameParts.slice(1).join(" ") || lastName;
  const accountData = {
    student: existingStudent._id,
    studentId: existingStudent.studentId,
    firstName: accountFirstName.trim(),
    lastName: accountLastName.trim(),
    email: accountEmail,
    password,
    department: (existingStudent.department || department).trim(),
    classDivision: (existingStudent.className || classDivision).trim(),
  };

  const account = await createStudentAccount(accountData);

  return {
    id: account._id,
    firstName: account.firstName,
    lastName: account.lastName,
    studentId: account.studentId,
    email: account.email,
    department: account.department,
    classDivision: account.classDivision,
    role: account.role,
    faceRegistered: Boolean(existingStudent.faceRegistered),
  };
}

async function createStudentAccount(accountData) {
  const session = await mongoose.startSession();

  try {
    try {
      session.startTransaction();
      const [account] = await StudentAccount.create([accountData], { session });
      await session.commitTransaction();
      return account;
    } catch (error) {
      if (session.inTransaction()) {
        await session.abortTransaction();
      }

      if (!isTransactionUnsupported(error)) {
        throw error;
      }
    }
  } finally {
    await session.endSession();
  }

  return StudentAccount.create(accountData);
}

function isTransactionUnsupported(error) {
  return /transaction|replica set|mongos/i.test(error.message || "");
}

async function loginStudent({ identifier, password }) {
  const normalizedIdentifier = identifier.trim();
  const query = normalizedIdentifier.includes("@")
    ? { email: normalizedIdentifier.toLowerCase() }
    : { studentId: normalizedIdentifier.toUpperCase() };

  const account = await StudentAccount.findOne(query).select("+password");
  if (!account) {
    const error = new Error("Student account not found. Please register first.");
    error.statusCode = 404;
    throw error;
  }

  if (!(await account.matchPassword(password))) {
    const error = new Error("Invalid Student ID/email or password");
    error.statusCode = 401;
    throw error;
  }

  const student = await Student.findById(account.student);
  if (!student) {
    const error = new Error("Linked student record was not found");
    error.statusCode = 500;
    throw error;
  }
  const token = generateStudentToken(account);

  return {
    token,
    student: {
      id: account._id,
      firstName: account.firstName,
      lastName: account.lastName,
      studentId: account.studentId,
      email: account.email,
      department: account.department,
      classDivision: account.classDivision,
      role: account.role,
      faceRegistered: student.faceRegistered === true,
    },
  };
}

module.exports = {
  registerTeacher,
  loginTeacher,
  getTeacherProfile,
  registerStudent,
  loginStudent,
};
