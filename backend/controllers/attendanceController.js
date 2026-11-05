/**
 * attendanceController.js
 * -----------------------------------------------------------------------
 * Handles attendance events received from the Python face engine.
 * -----------------------------------------------------------------------
 */

const Attendance = require("../models/Attendance");
const AttendanceSession = require("../models/AttendanceSession");
const Teacher = require("../models/Teacher");
const Schedule = require("../models/Schedule");
const Student = require("../models/Student");
const StudentAccount = require("../models/StudentAccount");
const {
  normalizeValue,
  getStudentClass,
  isStudentEligibleForSchedule,
} = require("../utils/classIdentity");
const { formatTime12, getScheduleTimeState } = require("../utils/scheduleTime");


// ---------------------------------------------------------------------
// POST /api/attendance
// ---------------------------------------------------------------------

const createAttendance = async (req, res, next) => {
  try {
    const {
      studentId: rawStudentId,
      name,
      date,
      time,
      status,
      trackId,
      similarity,
    } = req.body;

    // ---------------------------------------------------------
    // Basic validation
    // ---------------------------------------------------------

    const studentId = normalizeValue(rawStudentId);

    if (
      !studentId ||
      !name ||
      !date ||
      !time ||
      trackId === undefined ||
      similarity === undefined
    ) {
      return res.status(400).json({
        success: false,
        message:
          "studentId, name, date, time, trackId and similarity are required",
      });
    }

    // ---------------------------------------------------------
    // Do not accept UNKNOWN
    // ---------------------------------------------------------

    if (studentId === "UNKNOWN") {
      return res.status(400).json({
        success: false,
        message:
          "Unknown students cannot be marked present",
      });
    }

    // ---------------------------------------------------------
    // Find currently active attendance session
    // ---------------------------------------------------------

    const activeSession =
      await AttendanceSession.findOne({
        status: "live",
      }).sort({
        startedAt: -1,
      });

    if (!activeSession) {
      return res.status(400).json({
        success: false,
        message:
          "No active attendance session found. Start attendance first.",
      });
    }

    const schedule = await Schedule.findById(activeSession.scheduleId).lean();
    if (!schedule) {
      return res.status(409).json({
        success: false,
        message: "Attendance session has no valid schedule",
        reason: "invalid_session_schedule",
      });
    }

    if (
      normalizeValue(activeSession.classDivision) !== normalizeValue(schedule.classDivision) ||
      normalizeValue(activeSession.subject) !== normalizeValue(schedule.subject)
    ) {
      return res.status(409).json({
        success: false,
        message: "Attendance session does not match its schedule",
        reason: "session_schedule_mismatch",
      });
    }

    const student = await Student.findOne({ studentId }).lean();
    if (!student) {
      return res.status(404).json({
        success: false,
        message: "Recognized student was not found",
        reason: "student_not_found",
      });
    }

    const account = await StudentAccount.findOne({ student: student._id })
      .select("department classDivision")
      .lean();
    const eligibility = isStudentEligibleForSchedule(
      getStudentClass(student, account),
      schedule
    );

    if (!eligibility.eligible) {
      return res.status(403).json({
        success: false,
        message: "Attendance rejected: student does not belong to this class",
        reason: `class_mismatch:${eligibility.mismatches.join(",")}`,
      });
    }

    // ---------------------------------------------------------
    // Check whether student is already marked
    // in this attendance session
    // ---------------------------------------------------------

    const existingAttendance =
      await Attendance.findOne({
        studentId,
        sessionId: activeSession._id,
      });

    if (existingAttendance) {
      return res.status(200).json({
        success: true,
        alreadyMarked: true,
        message:
          "Attendance already marked for this student in this session",
        attendance: existingAttendance,
      });
    }

    // ---------------------------------------------------------
    // Create attendance record
    // ---------------------------------------------------------

    const attendance =
      await Attendance.create({
        studentId,
        name: student.name || `${student.firstName || ""} ${student.lastName || ""}`.trim(),

        teacherId:
          activeSession.teacherId,

        sessionId:
          activeSession._id,

        scheduleId:
          activeSession.scheduleId,

        subject:
          schedule.subject,

        classDivision:
          schedule.classDivision,

        department: schedule.department,
        academicYear: schedule.academicYear,
        division: schedule.division,

        date,
        time,

        status:
          status || "Present",

        trackId,
        similarity,
      });

    return res.status(201).json({
      success: true,
      alreadyMarked: false,
      message:
        "Attendance marked successfully",
      attendance,
    });

  } catch (error) {

    // ---------------------------------------------------------
    // Duplicate-key protection
    // ---------------------------------------------------------

    if (error.code === 11000) {
      return res.status(200).json({
        success: true,
        alreadyMarked: true,
        message:
          "Attendance already marked for this student in this session",
      });
    }

    next(error);
  }
};

// ---------------------------------------------------------------------
// GET /api/attendance
// ---------------------------------------------------------------------

const getAttendance = async (req, res, next) => {
  try {
    const { date } = req.query;
    const filter = date ? { date } : {};
    const attendance = await Attendance.find(filter)
      .sort({ date: -1, time: -1 });

    return res.status(200).json({
      success: true,
      count: attendance.length,
      attendance,
    });
  } catch (error) {
    next(error);
  }
};// ---------------------------------------------------------------------
// GET /api/teacher/attendance/history
// Get attendance history for the logged-in teacher
// ---------------------------------------------------------------------

const getTeacherAttendanceHistory = async (req, res, next) => {
  try {
    const Attendance = require("../models/Attendance");
    const Teacher = require("../models/Teacher");

    // req.user.id comes from the existing JWT authentication
    const teacher = await Teacher.findById(req.user.id).select("teacherId");

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    const { date, subject, classDivision } = req.query;

    // ---------------------------------------------------------------
    // Build filter
    // ---------------------------------------------------------------

    const filter = {
      teacherId: teacher.teacherId,
    };

    if (date) {
      filter.date = date;
    }

    if (subject) {
      filter.subject = subject;
    }

    if (classDivision) {
      filter.classDivision = classDivision;
    }

    // ---------------------------------------------------------------
    // Get attendance records
    // ---------------------------------------------------------------

    const attendanceRecords = await Attendance.find(filter)
      .sort({ date: -1, time: -1 })
      .lean();

    // ---------------------------------------------------------------
    // Group records into attendance sessions
    // ---------------------------------------------------------------

    const sessions = {};

    attendanceRecords.forEach((record) => {
      const key = [
        record.date,
        record.subject || "Unknown Subject",
        record.classDivision || "All Divisions",
        record.scheduleId || "no-schedule",
      ].join("_");

      if (!sessions[key]) {
        sessions[key] = {
          id: record.scheduleId || key,
          date: record.date,
          subject: record.subject || "Unknown Subject",
          classDivision: record.classDivision || "All Divisions",
          present: 0,
          absent: 0,
          status: "Completed",
        };
      }

      if (record.status === "Present") {
        sessions[key].present += 1;
      }

      if (record.status === "Absent") {
        sessions[key].absent += 1;
      }
    });

    // ---------------------------------------------------------------
    // Calculate attendance percentage
    // ---------------------------------------------------------------

    const history = Object.values(sessions).map((session) => {
      const total = session.present + session.absent;

      const percentage =
        total > 0
          ? ((session.present / total) * 100).toFixed(1)
          : "0.0";

      return {
        id: session.id,
        date: session.date,
        subject: session.subject,
        classDivision: session.classDivision,
        present: session.present,
        absent: session.absent,
        attendance: `${percentage}%`,
        status: session.status,
      };
    });

    return res.status(200).json({
      success: true,
      count: history.length,
      history,
    });
  } catch (error) {
    next(error);
  }
};

// ---------------------------------------------------------------------
// POST /api/attendance/sessions/start
// Start an attendance session for the logged-in teacher
// ---------------------------------------------------------------------

const startAttendanceSession = async (req, res, next) => {
  try {
    const { scheduleId } = req.body;

    // ---------------------------------------------------------------
    // Validate schedule ID
    // ---------------------------------------------------------------

    if (!scheduleId) {
      return res.status(400).json({
        success: false,
        message: "scheduleId is required",
      });
    }

    // ---------------------------------------------------------------
    // Get logged-in teacher
    // ---------------------------------------------------------------

    const teacher = await Teacher.findById(
      req.user.id
    ).select("teacherId");

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    // ---------------------------------------------------------------
    // Find schedule
    // ---------------------------------------------------------------

    const schedule = await Schedule.findById(scheduleId);

    if (!schedule) {
      return res.status(404).json({
        success: false,
        message: "Schedule not found",
      });
    }

    // ---------------------------------------------------------------
    // Security:
    // Make sure this schedule belongs to logged-in teacher
    // ---------------------------------------------------------------

    if (schedule.teacher !== teacher.teacherId) {
      return res.status(403).json({
        success: false,
        message:
          "You are not authorized to start attendance for this schedule",
      });
    }

    const existingSession = await AttendanceSession.findOne({
      scheduleId: schedule._id,
      teacherId: teacher.teacherId,
      date: schedule.date,
    });

    if (existingSession?.status === "live") {
      return res.status(200).json({
        success: true,
        message: "Attendance session is already live",
        session: existingSession,
      });
    }

    if (existingSession) {
      return res.status(409).json({
        success: false,
        message: "An attendance session already exists for this schedule.",
      });
    }

    const scheduleTimeState = getScheduleTimeState(schedule);
    if (scheduleTimeState.status === "invalid") {
      return res.status(400).json({
        success: false,
        message: "This schedule has an invalid lecture time.",
      });
    }
    if (scheduleTimeState.status === "upcoming") {
      return res.status(409).json({
        success: false,
        message: "Attendance cannot start before the scheduled lecture time.",
      });
    }
    if (scheduleTimeState.status === "expired") {
      return res.status(409).json({
        success: false,
        message: "Attendance cannot start after the scheduled lecture has ended.",
      });
    }

    // ---------------------------------------------------------------
    // Extract time
    // Existing Schedule stores:
    // "10:15 AM - 11:15 AM"
    // ---------------------------------------------------------------

    let startTime = "";
    let endTime = "";

    if (schedule.time && schedule.time.includes(" - ")) {
      const parts = schedule.time.split(" - ");

      startTime = formatTime12(parts[0]);
      endTime = formatTime12(parts[1]);
    }

    // ---------------------------------------------------------------
    // Create attendance session
    // ---------------------------------------------------------------

    const session = await AttendanceSession.create({
      teacherId: teacher.teacherId,
      scheduleId: schedule._id,
      subject: schedule.subject,
      classDivision: schedule.classDivision,
      department: schedule.department,
      academicYear: schedule.academicYear,
      division: schedule.division,
      room: schedule.room,
      date: schedule.date,
      startTime,
      endTime,
      status: "live",
      startedAt: new Date(),
    });

    return res.status(201).json({
      success: true,
      message: "Attendance session started successfully",
      session,
    });
  } catch (error) {
    next(error);
  }
};
// ---------------------------------------------------------------------
// GET /api/attendance/teacher/reports
// Teacher attendance reports
// ---------------------------------------------------------------------

const getTeacherReports = async (req, res, next) => {
  try {
    const teacher = await Teacher.findById(
      req.user.id
    ).select("teacherId");

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    const {
      classDivision,
      subject,
      startDate,
      endDate,
    } = req.query;

    // ---------------------------------------------------------------
    // Build AttendanceSession query
    // ---------------------------------------------------------------

    const sessionQuery = {
      teacherId: teacher.teacherId,
    };

    if (
      classDivision &&
      classDivision !== "All Classes"
    ) {
      sessionQuery.classDivision = classDivision;
    }

    if (
      subject &&
      subject !== "All Subjects"
    ) {
      sessionQuery.subject = subject;
    }

    if (startDate || endDate) {
      sessionQuery.date = {};

      if (startDate) {
        sessionQuery.date.$gte = startDate;
      }

      if (endDate) {
        sessionQuery.date.$lte = endDate;
      }
    }

    // ---------------------------------------------------------------
    // Find sessions
    // ---------------------------------------------------------------

    const sessions =
      await AttendanceSession.find(
        sessionQuery
      ).sort({
        date: 1,
      });

    if (sessions.length === 0) {
      return res.status(200).json({
        success: true,
        summary: {
          totalStudents: 0,
          averageAttendance: 0,
          highestAttendance: 0,
          lowAttendanceCount: 0,
        },
        subjectWise: [],
        lowAttendance: [],
      });
    }

    const sessionIds = sessions.map(
      (session) => session._id
    );

    // ---------------------------------------------------------------
    // Find attendance records for these sessions
    // ---------------------------------------------------------------

    const attendanceRecords =
      await Attendance.find({
        teacherId: teacher.teacherId,
        sessionId: {
          $in: sessionIds,
        },
        status: "Present",
      });

    // ---------------------------------------------------------------
    // Student attendance calculation
    // ---------------------------------------------------------------

    const studentMap = new Map();

    attendanceRecords.forEach((record) => {
      const studentId = record.studentId;

      if (!studentMap.has(studentId)) {
        studentMap.set(studentId, {
          studentId,
          name: record.name,
          present: 0,
          totalSessions: sessions.length,
        });
      }

      studentMap.get(studentId).present += 1;
    });

    const studentStats = Array.from(
      studentMap.values()
    ).map((student) => {
      const percentage =
        student.totalSessions > 0
          ? (student.present /
              student.totalSessions) *
            100
          : 0;

      return {
        ...student,
        attendance: Number(
          percentage.toFixed(1)
        ),
      };
    });

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------

    const totalStudents =
      studentStats.length;

    const averageAttendance =
      totalStudents > 0
        ? studentStats.reduce(
            (sum, student) =>
              sum + student.attendance,
            0
          ) / totalStudents
        : 0;

    const highestAttendance =
      totalStudents > 0
        ? Math.max(
            ...studentStats.map(
              (student) =>
                student.attendance
            )
          )
        : 0;

    const lowAttendance =
      studentStats
        .filter(
          (student) =>
            student.attendance < 75
        )
        .sort(
          (a, b) =>
            a.attendance -
            b.attendance
        );

    // ---------------------------------------------------------------
    // Subject-wise attendance
    // ---------------------------------------------------------------

    const subjectMap = new Map();

    sessions.forEach((session) => {
      const key = session.subject;

      if (!subjectMap.has(key)) {
        subjectMap.set(key, {
          subject: key,
          sessions: [],
        });
      }

      subjectMap
        .get(key)
        .sessions.push(session._id);
    });

    const subjectWise = [];

    for (const [
      subjectName,
      data,
    ] of subjectMap.entries()) {
      const subjectSessionIds =
        data.sessions;

      const subjectAttendance =
        attendanceRecords.filter(
          (record) =>
            subjectSessionIds.some(
              (id) =>
                id.toString() ===
                record.sessionId.toString()
            )
        );

      const uniqueStudents =
        new Set(
          subjectAttendance.map(
            (record) =>
              record.studentId
          )
        );

      if (uniqueStudents.size === 0) {
        subjectWise.push({
          subject: subjectName,
          value: 0,
        });

        continue;
      }

      const possibleAttendance =
        uniqueStudents.size *
        subjectSessionIds.length;

      const actualAttendance =
        subjectAttendance.length;

      const percentage =
        possibleAttendance > 0
          ? (actualAttendance /
              possibleAttendance) *
            100
          : 0;

      subjectWise.push({
        subject: subjectName,
        value: Number(
          percentage.toFixed(1)
        ),
      });
    }

    // ---------------------------------------------------------------
    // Response
    // ---------------------------------------------------------------

    return res.status(200).json({
      success: true,

      summary: {
        totalStudents,
        averageAttendance: Number(
          averageAttendance.toFixed(1)
        ),
        highestAttendance: Number(
          highestAttendance.toFixed(1)
        ),
        lowAttendanceCount:
          lowAttendance.length,
      },

      subjectWise,

      lowAttendance: lowAttendance.map(
        (student) => ({
          studentId:
            student.studentId,
          name: student.name,
          attendance:
            student.attendance,
          status:
            student.attendance < 75
              ? "Below Required"
              : "Good",
        })
      ),
    });
  } catch (error) {
    next(error);
  }
};
// ---------------------------------------------------------------------
// GET /api/attendance/teacher/reports/export
// Export teacher attendance report as CSV
// ---------------------------------------------------------------------

const exportTeacherReports = async (req, res, next) => {
  try {
    const teacher = await Teacher.findById(
      req.user.id
    ).select("teacherId");

    if (!teacher) {
      return res.status(404).json({
        success: false,
        message: "Teacher not found",
      });
    }

    const {
      classDivision,
      subject,
      startDate,
      endDate,
    } = req.query;

    // ---------------------------------------------------------------
    // Find teacher's attendance sessions
    // ---------------------------------------------------------------

    const sessionQuery = {
      teacherId: teacher.teacherId,
    };

    if (
      classDivision &&
      classDivision !== "All Classes"
    ) {
      sessionQuery.classDivision = classDivision;
    }

    if (
      subject &&
      subject !== "All Subjects"
    ) {
      sessionQuery.subject = subject;
    }

    if (startDate || endDate) {
      sessionQuery.date = {};

      if (startDate) {
        sessionQuery.date.$gte = startDate;
      }

      if (endDate) {
        sessionQuery.date.$lte = endDate;
      }
    }

    const sessions =
      await AttendanceSession.find(
        sessionQuery
      ).sort({
        date: 1,
      });

    const sessionIds = sessions.map(
      (session) => session._id
    );

    // ---------------------------------------------------------------
    // Find attendance records
    // ---------------------------------------------------------------

    const records =
      await Attendance.find({
        teacherId: teacher.teacherId,
        sessionId: {
          $in: sessionIds,
        },
      }).sort({
        date: 1,
        time: 1,
      });

    // ---------------------------------------------------------------
    // CSV header
    // ---------------------------------------------------------------

    const csvRows = [];

    csvRows.push([
      "Student ID",
      "Student Name",
      "Subject",
      "Class",
      "Date",
      "Time",
      "Status",
      "Similarity",
    ].join(","));

    // ---------------------------------------------------------------
    // CSV rows
    // ---------------------------------------------------------------

    records.forEach((record) => {
      const row = [
        record.studentId,
        record.name,
        record.subject,
        record.classDivision,
        record.date,
        record.time,
        record.status,
        record.similarity,
      ];

      csvRows.push(
        row
          .map((value) => {
            const text = String(
              value ?? ""
            );

            // Escape CSV special characters
            if (
              text.includes(",") ||
              text.includes('"') ||
              text.includes("\n")
            ) {
              return `"${text.replace(
                /"/g,
                '""'
              )}"`;
            }

            return text;
          })
          .join(",")
      );
    });

    const csv = csvRows.join("\n");

    // ---------------------------------------------------------------
    // Send CSV file
    // ---------------------------------------------------------------

    res.setHeader(
      "Content-Type",
      "text/csv; charset=utf-8"
    );

    res.setHeader(
      "Content-Disposition",
      'attachment; filename="attendance-report.csv"'
    );

    return res.status(200).send(csv);

  } catch (error) {
    next(error);
  }
};
module.exports = {
  createAttendance,
  getAttendance,
  getTeacherAttendanceHistory,
  startAttendanceSession,
  getTeacherReports,
  exportTeacherReports,
};