/**
 * authRoutes.js
 * -----------------------------------------------------------------------
 * Defines the teacher auth endpoints and wires up:
 *   validation -> validateMiddleware -> controller
 * for public routes, and:
 *   authMiddleware -> roleMiddleware -> controller
 * for protected routes.
 * -----------------------------------------------------------------------
 */

const express = require("express");
const { body } = require("express-validator");

const authController = require("../controllers/authController");
const validateRequest = require("../middleware/validateMiddleware");
const authMiddleware = require("../middleware/authMiddleware");
const roleMiddleware = require("../middleware/roleMiddleware");

const router = express.Router();

// ---------------------------------------------------------------------
// POST /api/auth/teacher/register
// ---------------------------------------------------------------------
const registerValidation = [
  body("firstName")
    .trim()
    .notEmpty()
    .withMessage("First name is required")
    .isLength({ min: 2, max: 50 })
    .withMessage("First name must be between 2 and 50 characters"),

  body("lastName").trim().notEmpty().withMessage("Last name is required"),

  body("teacherId").trim().notEmpty().withMessage("Teacher ID is required"),

  body("department").trim().notEmpty().withMessage("Department is required"),

  body("email").trim().notEmpty().withMessage("Email is required").isEmail().withMessage("Please provide a valid email address"),

  body("password").isLength({ min: 6 }).withMessage("Password must be at least 6 characters long"),
];

router.post("/teacher/register", registerValidation, validateRequest, authController.register);

// ---------------------------------------------------------------------
// POST /api/auth/teacher/login
// ---------------------------------------------------------------------
const loginValidation = [
  body("teacherId").trim().notEmpty().withMessage("Teacher ID is required"),
  body("password").notEmpty().withMessage("Password is required"),
];

router.post("/teacher/login", loginValidation, validateRequest, authController.login);

// ---------------------------------------------------------------------
// GET /api/auth/teacher/profile (protected)
// ---------------------------------------------------------------------
router.get("/teacher/profile", authMiddleware, roleMiddleware("teacher"), authController.getProfile);

module.exports = router;
