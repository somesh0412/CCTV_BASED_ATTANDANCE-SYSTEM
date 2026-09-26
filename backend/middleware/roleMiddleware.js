/**
 * roleMiddleware.js
 * -----------------------------------------------------------------------
 * Factory that returns a middleware restricting access to one or more
 * roles. Must run AFTER authMiddleware, since it relies on req.user
 * (set by authMiddleware) being populated already.
 *
 * Usage:
 *   router.get('/profile', authMiddleware, roleMiddleware('teacher'), handler)
 *
 * Supports multiple roles for future use, e.g. roleMiddleware('teacher', 'admin')
 * -----------------------------------------------------------------------
 */

const { errorResponse } = require("../utils/response");

function roleMiddleware(...allowedRoles) {
  return (req, res, next) => {
    if (!req.user) {
      // Should never happen if authMiddleware ran first, but guard anyway
      return errorResponse(res, 401, "Not authorized, no user context");
    }

    if (!allowedRoles.includes(req.user.role)) {
      return errorResponse(res, 403, "Forbidden: insufficient permissions");
    }

    return next();
  };
}

module.exports = roleMiddleware;
