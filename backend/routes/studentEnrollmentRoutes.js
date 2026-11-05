const express = require("express");
const multer = require("multer");

const authMiddleware = require("../middleware/authMiddleware");
const roleMiddleware = require("../middleware/roleMiddleware");
const {
  startEnrollment,
  processFrame,
  getProgress,
  cancelEnrollment,
} = require("../controllers/studentEnrollmentController");
const { errorResponse } = require("../utils/response");

const router = express.Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

function uploadFrame(req, res, next) {
  upload.single("frame")(req, res, (error) => {
    if (!error) {
      return next();
    }

    if (error.code === "LIMIT_FILE_SIZE") {
      return errorResponse(res, 413, "Frame image is too large");
    }
    return errorResponse(res, 400, "A valid multipart frame image is required");
  });
}

const studentOnly = [authMiddleware, roleMiddleware("student")];

router.post("/start", studentOnly, startEnrollment);
router.post("/process-frame", studentOnly, uploadFrame, processFrame);
router.get("/progress/:sessionId", studentOnly, getProgress);
router.post("/cancel/:sessionId", studentOnly, cancelEnrollment);

module.exports = router;
