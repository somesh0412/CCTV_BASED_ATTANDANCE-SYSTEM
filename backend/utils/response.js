/**
 * response.js
 * -----------------------------------------------------------------------
 * Standardized JSON response helpers so every endpoint returns the same
 * shape:
 *   success: { success: true, message, data }
 *   error:   { success: false, message }
 * -----------------------------------------------------------------------
 */

function successResponse(res, statusCode, message, data = {}) {
  return res.status(statusCode).json({
    success: true,
    message,
    data,
  });
}

function errorResponse(res, statusCode, message, errors = undefined) {
  const body = {
    success: false,
    message,
  };

  if (errors) {
    body.errors = errors;
  }

  return res.status(statusCode).json(body);
}

module.exports = { successResponse, errorResponse };
