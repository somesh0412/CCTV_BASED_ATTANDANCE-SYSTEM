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

/**
 * GET /api/auth/teacher/profile
 * Requires authMiddleware + roleMiddleware('teacher') to have run first.
 */
async function getProfile(req, res, next) {
  try {
    const teacher = await authService.getTeacherProfile(req.user.id);

    return res.status(200).json({
      success: true,
      teacher,
    });
  } catch (error) {
    return next(error);
  }
}

module.exports = { register, login, getProfile };
