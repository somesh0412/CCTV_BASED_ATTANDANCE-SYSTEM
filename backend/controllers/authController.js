/**
 * authController.js
 * -----------------------------------------------------------------------
 * Handles HTTP request/response concerns only. Delegates all actual
 * business logic to authService. Every handler is wrapped so thrown
 * errors are forwarded to the centralized error middleware via next(err).
 * -----------------------------------------------------------------------
 */

const authService = require("../services/authService");
const { successResponse } = require("../utils/response");
const Teacher = require("../models/Teacher");

/**
 * POST /api/auth/teacher/register
 */
async function register(req, res, next) {
  try {
    const { firstName, lastName, teacherId, department, email, password } = req.body;

    const teacher = await authService.registerTeacher({
      firstName,
      lastName,
      teacherId,
      department,
      email,
      password,
    });

    return successResponse(res, 201, "Teacher registered successfully", { teacher });
  } catch (error) {
    return next(error);
  }
}

/**
 * POST /api/auth/teacher/login
 */
async function login(req, res, next) {
  try {
    const { teacherId, password } = req.body;

    const { token, teacher } = await authService.loginTeacher({ teacherId, password });

    return successResponse(res, 200, "Login successful", { token, teacher });
  } catch (error) {
    return next(error);
  }
}


// ---------------------------------------------------------------------
// GET /api/auth/teacher/profile
// ---------------------------------------------------------------------

const getProfile = async (req, res, next) => {
  try {
    const teacher = await Teacher.findById(req.user.id).select(
      "-password"
    );

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    return res.status(200).json({
      success: true,
      data: {
        teacher,
      },
    });
  } catch (error) {
    next(error);
  }
};


// ---------------------------------------------------------------------
// PUT /api/auth/teacher/profile
// ---------------------------------------------------------------------

const updateProfile = async (req, res, next) => {
  try {
    const {
      firstName,
      lastName,
      email,
      department,
    } = req.body;

    const teacher = await Teacher.findById(
      req.user.id
    );

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    if (firstName !== undefined) {
      teacher.firstName = firstName.trim();
    }

    if (lastName !== undefined) {
      teacher.lastName = lastName.trim();
    }

    if (email !== undefined) {
      teacher.email = email.trim().toLowerCase();
    }

    if (department !== undefined) {
      teacher.department = department.trim();
    }

    await teacher.save();

    const updatedTeacher =
      teacher.toObject();

    delete updatedTeacher.password;

    return res.status(200).json({
      success: true,
      message: "Profile updated successfully",
      data: {
        teacher: updatedTeacher,
      },
    });
  } catch (error) {
    next(error);
  }
};

async function registerStudent(req, res, next) {
  try {
    const {
      firstName,
      lastName,
      studentId,
      email,
      password,
      department,
      classDivision,
    } = req.body;

    const student = await authService.registerStudent({
      firstName,
      lastName,
      studentId,
      email,
      password,
      department,
      classDivision,
    });

    return successResponse(res, 201, "Student account registered successfully", { student });
  } catch (error) {
    return next(error);
  }
}

async function loginStudent(req, res, next) {
  try {
    const { identifier, password } = req.body;
    const { token, student } = await authService.loginStudent({ identifier, password });

    return successResponse(res, 200, "Login successful", { token, student });
  } catch (error) {
    return next(error);
  }
}

module.exports = {
  register,
  login,
  getProfile,
  updateProfile,
  registerStudent,
  loginStudent,
};
