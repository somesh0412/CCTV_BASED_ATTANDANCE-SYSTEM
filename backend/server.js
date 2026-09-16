/**
 * server.js
 * -----------------------------------------------------------------------
 * Entry point. Responsibilities, in order:
 *   1. Load environment variables (via config/env.js, imported first)
 *   2. Connect to MongoDB
 *   3. Start the Express server
 * -----------------------------------------------------------------------
 */

const env = require("./config/env");
const connectDB = require("./config/db");
const app = require("./app");

async function startServer() {
  await connectDB();

  app.listen(env.PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`[server.js] Server running in ${env.NODE_ENV} mode on port ${env.PORT}`);
  });
}

startServer().catch((error) => {
  // eslint-disable-next-line no-console
  console.error("[server.js] Failed to start server:", error);
  process.exit(1);
});
