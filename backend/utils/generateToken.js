/**
 * generateToken.js
 * -----------------------------------------------------------------------
 * Small reusable helper to sign a JWT for an authenticated user.
 * Keeping this separate from the service/controller means the signing
 * logic (payload shape, expiry) lives in exactly one place.
 * -----------------------------------------------------------------------
 */

const jwt = require("jsonwebtoken");
const env = require("../config/env");

/**
 * Generate a signed JWT for a teacher.
 * @param {Object} teacher - Mongoose Teacher document (or plain object)
 * @returns {string} signed JWT
 */
function generateToken(user) {
  const payload = {
    id: user._id,
    teacherId: user.teacherId,
    studentId: user.studentId,
    role: user.role,
  };

  return jwt.sign(payload, env.JWT_SECRET, {
    expiresIn: env.JWT_EXPIRES_IN,
  });
}

module.exports = generateToken;
