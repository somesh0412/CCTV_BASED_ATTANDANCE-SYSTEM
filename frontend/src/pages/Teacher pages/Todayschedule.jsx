import React, { useEffect, useState } from "react";
import {
  Clock,
  MapPin,
  Plus,
  PlayCircle,
  Eye,
  X,
} from "lucide-react";
import api from "../../api/axious";
import "./TodaySchedule.css";

const SUBJECT_OPTIONS = [
  "Machine Learning",
  "Database Management System",
  "Computer Networks",
  "Operating Systems",
  "Software Engineering",
];

const CLASS_OPTIONS = [
  "FY CSD - A",
  "FY CSD - B",
  "FY CSD - C",
  "FY CSD - D",
  "SY CSD - A",
  "SY CSD - B",
  "SY CSD - C",
  "SY CSD - D",
  "TY CSD - A",
  "TY CSD - B",
  "TY CSD - C",
  "TY CSD - D",
];

const TIME_HOURS = Array.from({ length: 12 }, (_, index) =>
  String(index + 1).padStart(2, "0")
);
const TIME_MINUTES = Array.from({ length: 60 }, (_, index) =>
  String(index).padStart(2, "0")
);
const MERIDIEMS = ["AM", "PM"];

const TODAY_LABEL = new Date().toLocaleDateString("en-US", {
  weekday: "long",
  day: "numeric",
  month: "long",
  year: "numeric",
});

function formatTime(value) {
  if (!value) return "";

  const match = String(value).trim().match(/^(\d{1,2}):(\d{2})(?:\s*(AM|PM))?$/i);
  if (!match) return "";

  let hour = Number(match[1]);
  const minute = match[2];
  const meridiem = match[3]?.toUpperCase();
  if (meridiem) {
    if (hour === 12) hour = 0;
    if (meridiem === "PM") hour += 12;
  }
  return `${String(hour % 12 || 12).padStart(2, "0")}:${minute} ${
    hour >= 12 ? "PM" : "AM"
  }`;
}

function to24Hour(hour, minute, meridiem) {
  let hour24 = Number(hour);
  if (meridiem === "AM" && hour24 === 12) hour24 = 0;
  if (meridiem === "PM" && hour24 !== 12) hour24 += 12;
  return `${String(hour24).padStart(2, "0")}:${minute}`;
}

function getScheduleStatus(item, now = new Date()) {
  const [startTime, endTime] = String(item.time || `${item.startTime} - ${item.endTime}`)
    .split(" - ");
  const parseTime = (value) => {
    const match = String(value || "").trim().match(/^(\d{1,2}):(\d{2})(?:\s*(AM|PM))?$/i);
    if (!match) return null;
    let hour = Number(match[1]);
    const minute = Number(match[2]);
    const meridiem = match[3]?.toUpperCase();
    if (minute > 59 || hour > (meridiem ? 12 : 23)) return null;
    if (meridiem) {
      if (hour === 12) hour = 0;
      if (meridiem === "PM") hour += 12;
    }
    return hour * 60 + minute;
  };

  const startMinutes = parseTime(startTime);
  const endMinutes = parseTime(endTime);
  if (!item.date || startMinutes === null || endMinutes === null) return "invalid";

  const dateParts = String(item.date).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!dateParts) return "invalid";
  const dateSerial =
    Date.UTC(Number(dateParts[1]), Number(dateParts[2]) - 1, Number(dateParts[3])) /
    86400000;
  const startSerial = dateSerial * 1440 + startMinutes;
  const endSerial = dateSerial * 1440 + (endMinutes <= startMinutes ? endMinutes + 1440 : endMinutes);
  const indiaParts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(now);
  const values = Object.fromEntries(
    indiaParts
      .filter(({ type }) => type !== "literal")
      .map(({ type, value }) => [type, Number(value)])
  );
  const nowSerial =
    Date.UTC(values.year, values.month - 1, values.day) / 86400000 * 1440 +
    values.hour * 60 +
    values.minute;

  if (nowSerial < startSerial) return "upcoming";
  if (nowSerial >= endSerial) return "expired";
  return "active";
}


/* =========================================================
   ADD CLASS MODAL
========================================================= */

function AddClassModal({ open, onClose, onAdd }) {
  const [form, setForm] = useState({
    subject: "",
    classDivision: "",
    startHour: "",
    startMinute: "",
    startMeridiem: "",
    endHour: "",
    endMinute: "",
    endMeridiem: "",
    room: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  if (!open) return null;

  const handleChange = (field) => (e) => {
    setForm((prev) => ({
      ...prev,
      [field]: e.target.value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (
      !form.subject ||
      !form.classDivision ||
      !form.startHour ||
      !form.startMinute ||
      !form.startMeridiem ||
      !form.endHour ||
      !form.endMinute ||
      !form.endMeridiem ||
      !form.room
    ) {
      setError("Please fill all fields.");
      return;
    }

    const startTime = to24Hour(form.startHour, form.startMinute, form.startMeridiem);
    const endTime = to24Hour(form.endHour, form.endMinute, form.endMeridiem);
    if (startTime === endTime) {
      setError("End time must be later than start time.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const response = await api.post("/schedules/teacher", {
        subject: form.subject,
        classDivision: form.classDivision,
        startTime,
        endTime,
        room: form.room,
      });

      onAdd(response.data.schedule);

      setForm({
        subject: "",
        classDivision: "",
        startHour: "",
        startMinute: "",
        startMeridiem: "",
        endHour: "",
        endMinute: "",
        endMeridiem: "",
        room: "",
      });

      onClose();
    } catch (err) {
      console.error("Failed to create schedule:", err);

      setError(
        err.response?.data?.message ||
          "Failed to create class."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="ts-modal-overlay"
      onClick={onClose}
    >
      <div
        className="ts-modal"
        onClick={(e) => e.stopPropagation()}
      >

        <div className="ts-modal__header">

          <h3>Add Class</h3>

          <button
            type="button"
            className="ts-modal__close"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={18} />
          </button>

        </div>


        <form
          className="ts-modal__form"
          onSubmit={handleSubmit}
        >

          {/* Subject */}

          <label className="ts-field">

            <span>Subject</span>

            <select
              value={form.subject}
              onChange={handleChange("subject")}
            >
              <option value="" disabled>
                Select Subject
              </option>

              {SUBJECT_OPTIONS.map((subject) => (
                <option
                  key={subject}
                  value={subject}
                >
                  {subject}
                </option>
              ))}

            </select>

          </label>


          {/* Class */}

          <label className="ts-field">

            <span>Class / Division</span>

            <select
              value={form.classDivision}
              onChange={handleChange("classDivision")}
            >
              <option value="" disabled>
                Select Class
              </option>

              {CLASS_OPTIONS.map((className) => (
                <option
                  key={className}
                  value={className}
                >
                  {className}
                </option>
              ))}

            </select>

          </label>


          {/* Time */}

          <div className="ts-modal__row ts-time-fields">

            <label className="ts-field ts-time-field">

              <span>Start Time</span>

              <div className="ts-time-picker">
                <select value={form.startHour} onChange={handleChange("startHour")} aria-label="Start hour">
                  <option value="" disabled>HH</option>
                  {TIME_HOURS.map((hour) => <option key={hour} value={hour}>{hour}</option>)}
                </select>
                <select value={form.startMinute} onChange={handleChange("startMinute")} aria-label="Start minute">
                  <option value="" disabled>MM</option>
                  {TIME_MINUTES.map((minute) => <option key={minute} value={minute}>{minute}</option>)}
                </select>
                <select value={form.startMeridiem} onChange={handleChange("startMeridiem")} aria-label="Start AM or PM">
                  <option value="" disabled>AM/PM</option>
                  {MERIDIEMS.map((meridiem) => <option key={meridiem} value={meridiem}>{meridiem}</option>)}
                </select>
              </div>

            </label>


            <label className="ts-field ts-time-field">

              <span>End Time</span>

              <div className="ts-time-picker">
                <select value={form.endHour} onChange={handleChange("endHour")} aria-label="End hour">
                  <option value="" disabled>HH</option>
                  {TIME_HOURS.map((hour) => <option key={hour} value={hour}>{hour}</option>)}
                </select>
                <select value={form.endMinute} onChange={handleChange("endMinute")} aria-label="End minute">
                  <option value="" disabled>MM</option>
                  {TIME_MINUTES.map((minute) => <option key={minute} value={minute}>{minute}</option>)}
                </select>
                <select value={form.endMeridiem} onChange={handleChange("endMeridiem")} aria-label="End AM or PM">
                  <option value="" disabled>AM/PM</option>
                  {MERIDIEMS.map((meridiem) => <option key={meridiem} value={meridiem}>{meridiem}</option>)}
                </select>
              </div>

            </label>

          </div>


          {/* Classroom */}

          <label className="ts-field">

            <span>Classroom</span>

            <input
              type="text"
              placeholder="e.g. Room 302"
              value={form.room}
              onChange={handleChange("room")}
            />

          </label>


          {error && (
            <p
              style={{
                margin: "0",
                color: "#dc2626",
                fontSize: "0.8rem",
              }}
            >
              {error}
            </p>
          )}


          {/* Buttons */}

          <div className="ts-modal__actions">

            <button
              type="button"
              className="ts-btn ts-btn--outline"
              onClick={onClose}
              disabled={loading}
            >
              Cancel
            </button>

            <button
              type="submit"
              className="ts-btn ts-btn--green"
              disabled={loading}
            >
              {loading ? "Adding..." : "Add Class"}
            </button>

          </div>

        </form>

      </div>
    </div>
  );
}


/* =========================================================
   SCHEDULE CARD
========================================================= */

function ScheduleCard({
  item,
  onStartAttendance,
  sessionStarted,
  currentTime,
}) {
  const scheduleStatus = getScheduleStatus(item, currentTime);
  const isActive = scheduleStatus === "active";
  const statusLabel = {
    upcoming: "Upcoming",
    active: "Active",
    expired: "Expired",
    invalid: "Unavailable",
  }[scheduleStatus];

  return (
    <div
      className={`ts-item ${
        isActive ? "ts-item--live" : ""
      }`}
    >

      {/* Time */}

      <div className="ts-item__time">

        <Clock size={16} />

        <span>
          {formatTime(item.startTime)}
          {" – "}
          {formatTime(item.endTime)}
        </span>

      </div>


      {/* Subject */}

      <div className="ts-item__info">

        <h4>{item.subject}</h4>

        <p>

          {item.classDivision}

          <span className="ts-dot">
            &bull;
          </span>

          <MapPin size={13} />

          {item.room}

        </p>

      </div>


      {/* Status */}

      <div className="ts-item__status">

        <span
          className={`ts-badge ${
            isActive
              ? "ts-badge--live"
              : "ts-badge--upcoming"
          }`}
        >
          {statusLabel}
        </span>

      </div>


      {/* Action */}

      <div className="ts-item__action">

        <button
          type="button"
          className="ts-btn ts-btn--green"
          disabled={!isActive || sessionStarted}
          onClick={() =>
            isActive && onStartAttendance(item)
          }
        >

          <PlayCircle size={16} />

          {sessionStarted
            ? "Session Started"
            : "Start Attendance"}

        </button>

      </div>

    </div>
  );
}


/* =========================================================
   MAIN COMPONENT
========================================================= */

export default function TodaySchedule({
  onStartAttendance,
}) {

  const [schedule, setSchedule] = useState([]);

  const [loading, setLoading] = useState(true);

  const [error, setError] = useState("");

  const [modalOpen, setModalOpen] =
    useState(false);

  const [startedSessions, setStartedSessions] =
    useState({});
  const [currentTime, setCurrentTime] = useState(() => new Date());


  /* =======================================================
     GET TODAY'S SCHEDULE
  ======================================================= */

  const fetchSchedule = async () => {

    try {

      setLoading(true);
      setError("");

      const response = await api.get(
        "/schedules/teacher/today"
      );

      setSchedule(
        response.data.schedules || []
      );

    } catch (err) {

      console.error(
        "Failed to fetch schedule:",
        err
      );

      setError(
        err.response?.data?.message ||
          "Failed to load today's schedule."
      );

    } finally {

      setLoading(false);

    }
  };


  useEffect(() => {
    fetchSchedule();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setCurrentTime(new Date());
    }, 30000);

    return () => window.clearInterval(timer);
  }, []);


  /* =======================================================
     ADD CLASS
  ======================================================= */

  const handleAddClass = (newItem) => {

    setSchedule((prev) => [
      ...prev,
      newItem,
    ]);

  };


  /* =======================================================
     START ATTENDANCE
  ======================================================= */

  const handleStart = (item) => {

    setStartedSessions((prev) => ({
      ...prev,
      [item.id]: true,
    }));

    /*
      Send selected schedule to TeacherDashboard.
      TeacherDashboard will open Live Attendance.
    */

    if (onStartAttendance) {
      onStartAttendance(item);
    }

  };


  return (
    <div className="ts-page">

      {/* HEADER */}

      <div className="ts-header">

        <div>

          <h1>
            Today's Schedule
          </h1>

          <p>
            View and manage your classes for today.
          </p>

          <span className="ts-today">
            {TODAY_LABEL}
          </span>

        </div>


        <button
          type="button"
          className="ts-btn ts-btn--green"
          onClick={() =>
            setModalOpen(true)
          }
        >

          <Plus size={16} />

          Add Class

        </button>

      </div>


      {/* SCHEDULE */}

      <div className="ts-card">

        {loading ? (

          <p className="ts-empty">
            Loading today's schedule...
          </p>

        ) : error ? (

          <p className="ts-empty">
            {error}
          </p>

        ) : (

          <div className="ts-list">

            {schedule.map((item) => (

              <ScheduleCard
                key={item.id}
                item={item}
                onStartAttendance={handleStart}
                sessionStarted={
                  !!startedSessions[item.id]
                }
                currentTime={currentTime}
              />

            ))}


            {schedule.length === 0 && (

              <p className="ts-empty">
                No classes scheduled for today yet.
              </p>

            )}

          </div>

        )}

      </div>


      {/* MODAL */}

      <AddClassModal
        open={modalOpen}
        onClose={() =>
          setModalOpen(false)
        }
        onAdd={handleAddClass}
      />

    </div>
  );
}