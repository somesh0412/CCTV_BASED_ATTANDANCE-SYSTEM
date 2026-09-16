/**
 * validateMiddleware.js
 * -----------------------------------------------------------------------
 * Generic middleware that checks the results collected by express-validator
 * validation chains (defined in the routes file) and short-circuits the
 * request with a 400 response if any validation failed.
 * -----------------------------------------------------------------------
 */

const { validationResult } = require("express-validator");
const { errorResponse } = require("../utils/response");

function validateRequest(req, res, next) {
  const errors = validationResult(req);

  if (!errors.isEmpty()) {
    const formattedErrors = errors.array().map((err) => ({
      field: err.path,
      message: err.msg,
    }));

    return errorResponse(res, 400, "Validation failed", formattedErrors);
  }

  return next();
}

module.exports = validateRequest;
