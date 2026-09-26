/**
 * errorMiddleware.js
 * -----------------------------------------------------------------------
 * Centralized error handling for the app:
 *  - notFound: catches requests to undefined routes -> 404
 *  - errorHandler: catches everything passed to next(err) from
 *    controllers/services (including Mongoose validation/duplicate-key
 *    errors) and returns a consistent JSON error response.
 * -----------------------------------------------------------------------
 */

const env = require("../config/env");

function notFound(req, res, next) {
  const error = new Error(`Route not found - ${req.originalUrl}`);
  error.statusCode = 404;
  next(error);
}

// eslint-disable-next-line no-unused-vars
function errorHandler(err, req, res, next) {
  let statusCode = err.statusCode || 500;
  let message = err.message || "Server error";

  // Mongoose duplicate key error (e.g. teacherId or email already exists)
  if (err.code === 11000) {
    statusCode = 409;
    const field = Object.keys(err.keyValue || {})[0] || "field";
    message = `${field} already exists`;
  }

  // Mongoose validation error
  if (err.name === "ValidationError") {
    statusCode = 400;
    message = Object.values(err.errors)
      .map((e) => e.message)
      .join(", ");
  }

  // Mongoose invalid ObjectId / CastError
  if (err.name === "CastError") {
    statusCode = 400;
    message = `Invalid value for ${err.path}`;
  }

  res.status(statusCode).json({
    success: false,
    message,
    ...(env.NODE_ENV === "development" ? { stack: err.stack } : {}),
  });
}

module.exports = { notFound, errorHandler };
