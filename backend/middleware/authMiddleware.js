/**
 * authMiddleware.js
 * -----------------------------------------------------------------------
 * Verifies the JWT sent in the Authorization header:
 *    Authorization: Bearer <token>
 *
 * On success, attaches the decoded payload to req.user and calls next().
 * On failure (missing / malformed / invalid / expired token), responds
 * with HTTP 401.
 * -----------------------------------------------------------------------
 */

const jwt = require("jsonwebtoken");
const env = require("../config/env");
const { errorResponse } = require("../utils/response");

async function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return errorResponse(res, 401, "Not authorized, no token provided");
  }

  const token = authHeader.split(" ")[1];

  if (!token) {
    return errorResponse(res, 401, "Not authorized, no token provided");
  }

  try {
    const decoded = jwt.verify(token, env.JWT_SECRET);

    // decoded: { id, teacherId, role, iat, exp }
    req.user = decoded;

    return next();
  } catch (error) {
    if (error.name === "TokenExpiredError") {
      return errorResponse(res, 401, "Token expired, please log in again");
    }
    return errorResponse(res, 401, "Not authorized, invalid token");
  }
}

module.exports = authMiddleware;
