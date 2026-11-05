const mongoose = require("mongoose");
const Student = require("../models/Student");
const StudentAccount = require("../models/StudentAccount");
const StudentEnrollmentSession = require("../models/StudentEnrollmentSession");
const env = require("../config/env");
const { successResponse, errorResponse } = require("../utils/response");

const MIN_EMBEDDINGS = 25;
const EMBEDDING_DIMENSION = 512;
const FLASK_TIMEOUT_MS = 15_000;

function createHttpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

function getFlaskErrorMessage(payload, status) {
  if (payload && typeof payload.error === "string") {
    return payload.error;
  }
  return `Face enrollment service returned HTTP ${status}`;
}

async function callFlask(path, options = {}) {
  if (!env.FACE_ENROLLMENT_API_SECRET) {
    throw createHttpError(
      503,
      "Face enrollment service is not configured"
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FLASK_TIMEOUT_MS);

  try {
    const baseUrl = env.FACE_ENROLLMENT_API_URL.replace(/\/+$/, "");
    const response = await fetch(`${baseUrl}${path}`, {
      ...options,
      headers: {
        "X-Enrollment-Secret": env.FACE_ENROLLMENT_API_SECRET,
        ...(options.headers || {}),
      },
      signal: controller.signal,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : null;

    if (!response.ok) {
      const statusCode =
        response.status === 401 || response.status === 503
          ? 502
          : response.status === 404
            ? 409
            : response.status >= 400 && response.status < 500
              ? response.status
              : 502;
      throw createHttpError(
        statusCode,
        getFlaskErrorMessage(payload, response.status)
      );
    }

    return payload;
  } catch (error) {
    if (error.name === "AbortError") {
      throw createHttpError(504, "Face enrollment service timed out");
    }
    if (error.statusCode) {
      throw error;
    }
    throw createHttpError(503, "Face enrollment service is unavailable");
  } finally {
    clearTimeout(timeout);
  }
}

async function getAuthenticatedStudent(req) {
  if (!mongoose.Types.ObjectId.isValid(req.user.id)) {
    throw createHttpError(401, "Invalid student authentication context");
  }

  const account = await StudentAccount.findById(req.user.id).select("student");
  if (!account) {
    throw createHttpError(401, "Student account not found");
  }

  const student = await Student.findById(account.student);
  if (!student) {
    throw createHttpError(404, "Linked student record was not found");
  }

  return student;
}

function validateEmbeddings(embeddings) {
  if (!Array.isArray(embeddings) || embeddings.length < MIN_EMBEDDINGS) {
    throw createHttpError(
      502,
      `Enrollment service returned fewer than ${MIN_EMBEDDINGS} embeddings`
    );
  }

  const valid = embeddings.every(
    (embedding) =>
      Array.isArray(embedding) &&
      embedding.length === EMBEDDING_DIMENSION &&
      embedding.every(
        (value) => typeof value === "number" && Number.isFinite(value)
      )
  );

  if (!valid) {
    throw createHttpError(
      502,
      `Enrollment service returned an invalid embedding; each embedding must contain exactly ${EMBEDDING_DIMENSION} finite numbers`
    );
  }

  return embeddings;
}

function getProgressStatus(payload) {
  return payload?.progress?.current_status;
}

async function startEnrollment(req, res, next) {
  try {
    const student = await getAuthenticatedStudent(req);

    if (
      student.faceRegistered === true ||
      (Array.isArray(student.faceEmbeddings) && student.faceEmbeddings.length > 0)
    ) {
      return errorResponse(res, 409, "Student face is already enrolled");
    }

    const activeSession = await StudentEnrollmentSession.findOne({
      student: student._id,
      status: "active",
    });
    if (activeSession) {
      return errorResponse(res, 409, "Student enrollment is already in progress");
    }

    const flaskResponse = await callFlask("/enrollment/start", { method: "POST" });
    if (!flaskResponse?.session_id || !flaskResponse.progress) {
      throw createHttpError(502, "Face enrollment service returned an invalid start response");
    }

    let session;
    try {
      session = await StudentEnrollmentSession.create({
        student: student._id,
        flaskSessionId: flaskResponse.session_id,
        lastProgress: flaskResponse.progress,
      });
    } catch (error) {
      try {
        await callFlask(`/enrollment/cancel/${encodeURIComponent(flaskResponse.session_id)}`, {
          method: "POST",
        });
      } catch {
        // The original persistence error is more useful to the caller.
      }
      if (error.code === 11000) {
        return errorResponse(res, 409, "Student enrollment is already in progress");
      }
      throw error;
    }

    return successResponse(res, 201, "Enrollment session started", {
      sessionId: session._id,
      flaskSessionId: session.flaskSessionId,
      progress: flaskResponse.progress,
    });
  } catch (error) {
    return next(error);
  }
}

async function processFrame(req, res, next) {
  try {
    const student = await getAuthenticatedStudent(req);
    const flaskSessionId = req.body.session_id;
    if (!flaskSessionId || !req.file) {
      return errorResponse(res, 400, "session_id and a frame image are required");
    }

    const session = await StudentEnrollmentSession.findOne({
      student: student._id,
      flaskSessionId,
      status: "active",
    });
    if (!session) {
      return errorResponse(res, 409, "Enrollment session is invalid or no longer active");
    }

    const form = new FormData();
    form.set("session_id", flaskSessionId);
    form.set("frame", new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname);

    let flaskResponse;
    try {
      flaskResponse = await callFlask("/enrollment/process-frame", {
        method: "POST",
        body: form,
      });
    } catch (error) {
      if (error.statusCode === 408 || error.statusCode === 409) {
        session.status = "failed";
        await session.save();
      }
      throw error;
    }
    const status = getProgressStatus(flaskResponse);

    session.lastProgress = flaskResponse?.progress || null;

    if (status === "completed") {
      if (flaskResponse.embeddings === undefined) {
        session.status = "failed";
        await session.save();
        throw createHttpError(
          502,
          "Enrollment service completed without returning embeddings"
        );
      }

      let embeddings;
      try {
        embeddings = validateEmbeddings(flaskResponse.embeddings);
      } catch (error) {
        session.status = "failed";
        await session.save();
        throw error;
      }

      const updatedStudent = await Student.findOneAndUpdate(
        {
          _id: student._id,
          faceRegistered: { $ne: true },
          $or: [{ faceEmbeddings: { $exists: false } }, { faceEmbeddings: { $size: 0 } }],
        },
        { $set: { faceRegistered: true, faceEmbeddings: embeddings } },
        { new: true, runValidators: true }
      );

      if (!updatedStudent) {
        session.status = "completed";
        await session.save();
        return errorResponse(res, 409, "Student face is already enrolled");
      }

      session.status = "completed";
      session.embeddingsStored = true;
      session.completedAt = new Date();
    }

    await session.save();
    return successResponse(res, 200, "Enrollment frame processed", {
      sessionId: session._id,
      ...flaskResponse,
    });
  } catch (error) {
    return next(error);
  }
}

async function getProgress(req, res, next) {
  try {
    const student = await getAuthenticatedStudent(req);
    const session = await StudentEnrollmentSession.findOne({
      student: student._id,
      flaskSessionId: req.params.sessionId,
    });
    if (!session) {
      return errorResponse(res, 404, "Enrollment session not found");
    }

    const flaskResponse = await callFlask(
      `/enrollment/progress/${encodeURIComponent(session.flaskSessionId)}`
    );
    session.lastProgress = flaskResponse?.progress || null;
    if (getProgressStatus(flaskResponse) === "completed" && session.status === "active") {
      session.status = "completed";
      session.completedAt = new Date();
    }
    await session.save();

    return successResponse(res, 200, "Enrollment progress retrieved", {
      sessionId: session._id,
      ...flaskResponse,
    });
  } catch (error) {
    return next(error);
  }
}

async function cancelEnrollment(req, res, next) {
  try {
    const student = await getAuthenticatedStudent(req);
    const session = await StudentEnrollmentSession.findOne({
      student: student._id,
      flaskSessionId: req.params.sessionId,
      status: "active",
    });
    if (!session) {
      return errorResponse(res, 409, "Enrollment session is invalid or no longer active");
    }

    const flaskResponse = await callFlask(
      `/enrollment/cancel/${encodeURIComponent(session.flaskSessionId)}`,
      { method: "POST" }
    );
    session.status = "cancelled";
    session.lastProgress = flaskResponse?.progress || null;
    await session.save();

    return successResponse(res, 200, "Enrollment cancelled", {
      sessionId: session._id,
      ...flaskResponse,
    });
  } catch (error) {
    return next(error);
  }
}

module.exports = {
  startEnrollment,
  processFrame,
  getProgress,
  cancelEnrollment,
};
