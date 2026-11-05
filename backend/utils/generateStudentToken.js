const jwt = require("jsonwebtoken");
const env = require("../config/env");

function generateStudentToken(studentAccount) {
  return jwt.sign(
    {
      id: studentAccount._id,
      studentId: studentAccount.studentId,
      role: studentAccount.role,
    },
    env.JWT_SECRET,
    { expiresIn: env.JWT_EXPIRES_IN }
  );
}

module.exports = generateStudentToken;
