const Attendance = require("../models/Attendance");
const AttendanceSession = require("../models/AttendanceSession");
const Schedule = require("../models/Schedule");
const Student = require("../models/Student");
const StudentAccount = require("../models/StudentAccount");
const { getStudentClass, isStudentEligibleForSchedule } = require("../utils/classIdentity");

async function getStudentDashboard(req, res, next) {
  try {
    const account = await StudentAccount.findById(req.user.id).select(
      "student studentId firstName lastName email department classDivision role"
    );

    if (!account || account.role !== "student") {
      return res.status(404).json({
        success: false,
        message: "Student account not found",
      });
    }

    const student = await Student.findById(account.student).select(
      "studentId firstName lastName name email department className faceRegistered updatedAt"
    );

    if (!student) {
      return res.status(404).json({
        success: false,
        message: "Linked student record was not found",
      });
    }

    const studentId = student.studentId || account.studentId;
    const today = new Date().toISOString().split("T")[0];
    const studentClass = getStudentClass(student, account);

    const [allSessions, schedules] = await Promise.all([
      AttendanceSession.find({ date: { $lte: today } })
        .populate("scheduleId")
        .select("scheduleId subject classDivision room date startTime endTime status")
        .sort({ date: -1, startTime: -1 })
        .lean(),
      Schedule.find({
        date: today,
      }).select("subject time room classDivision department academicYear division date status").sort({ time: 1 }).lean(),
    ]);

    const eligibleSessions = allSessions.filter((session) => {
      const schedule = session.scheduleId;
      return schedule && isStudentEligibleForSchedule(studentClass, schedule).eligible;
    });
    const eligibleSessionIds = eligibleSessions.map((session) => session._id);
    const eligibleSchedules = schedules.filter((schedule) =>
      isStudentEligibleForSchedule(studentClass, schedule).eligible
    );
    const attendance = await Attendance.find({
      studentId,
      sessionId: { $in: eligibleSessionIds },
    })
      .select("sessionId date time subject classDivision status")
      .sort({ date: -1, time: -1 })
      .lean();

    const attendedBySession = new Set(
      attendance
        .filter((record) => record.status === "Present")
        .map((record) => String(record.sessionId))
    );
    const absentBySession = new Set(
      attendance
        .filter((record) => record.status === "Absent")
        .map((record) => String(record.sessionId))
    );
    const subjectStats = new Map();

    eligibleSessions.forEach((session) => {
      const key = String(session._id);
      const subject = subjectStats.get(session.subject) || {
        name: session.subject,
        total: 0,
        attended: 0,
      };

      subject.total += 1;
      if (attendedBySession.has(key)) {
        subject.attended += 1;
      }
      subjectStats.set(session.subject, subject);
    });

    const attended = [...subjectStats.values()].reduce(
      (total, subject) => total + subject.attended,
      0
    );
    const total = eligibleSessions.length;
    const missed = absentBySession.size;

    return res.status(200).json({
      success: true,
      data: {
        student: {
          id: student._id,
          firstName: student.firstName || account.firstName,
          lastName: student.lastName || account.lastName,
          studentId,
          email: student.email || account.email,
          department: student.department || account.department,
          classDivision: account.classDivision || student.className,
          faceRegistered: student.faceRegistered === true,
          faceUpdatedAt: student.updatedAt || null,
        },
        attendance: {
          total,
          attended,
          missed,
          percentage: total ? Number(((attended / total) * 100).toFixed(1)) : null,
          subjects: [...subjectStats.values()].map((subject) => ({
            ...subject,
            percentage: subject.total
              ? Number(((subject.attended / subject.total) * 100).toFixed(1))
              : null,
          })),
          records: attendance.map((record) => ({
            date: record.date,
            time: record.time,
            subject: record.subject,
            classDivision: record.classDivision,
            status: record.status,
          })),
        },
        schedules: eligibleSchedules.map((schedule) => ({
          id: schedule._id,
          subject: schedule.subject,
          time: schedule.time,
          room: schedule.room,
          classDivision: schedule.classDivision,
          date: schedule.date,
          status: schedule.status,
        })),
      },
    });
  } catch (error) {
    return next(error);
  }
}

module.exports = { getStudentDashboard };
