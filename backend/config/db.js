/**
 * db.js
 * -----------------------------------------------------------------------
 * Connects to MongoDB using Mongoose. Exits the process on failure so the
 * server never runs in a "half-connected" state.
 * -----------------------------------------------------------------------
 */

const mongoose = require("mongoose");
const env = require("./env");

const connectDB = async () => {
  try {
    await mongoose.connect(env.MONGO_URI);
    // eslint-disable-next-line no-console
    console.log(`[db.js] MongoDB connected: ${mongoose.connection.host}`);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error(`[db.js] MongoDB connection failed: ${error.message}`);
    process.exit(1);
  }
};

module.exports = connectDB;
