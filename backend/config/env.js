/**
 * env.js
 * -----------------------------------------------------------------------
 * Loads environment variables from .env and validates that all required
 * variables are present before the app starts. Fails fast with a clear
 * error message instead of letting the app crash later with a confusing
 * stack trace (e.g. "Cannot read property 'sign' of undefined").
 * -----------------------------------------------------------------------
 */

require("dotenv").config();

const REQUIRED_ENV_VARS = ["PORT", "MONGO_URI", "JWT_SECRET", "JWT_EXPIRES_IN"];

function validateEnv() {
  const missing = REQUIRED_ENV_VARS.filter((key) => !process.env[key]);

  if (missing.length > 0) {
    // eslint-disable-next-line no-console
    console.error(
      `[env.js] Missing required environment variable(s): ${missing.join(", ")}\n` +
        "Please create a .env file based on .env example and set these values."
    );
    process.exit(1);
  }
}

validateEnv();

const env = {
  PORT: process.env.PORT,
  MONGO_URI: process.env.MONGO_URI,
  JWT_SECRET: process.env.JWT_SECRET,
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN,
  NODE_ENV: process.env.NODE_ENV || "development",
};

module.exports = env;
