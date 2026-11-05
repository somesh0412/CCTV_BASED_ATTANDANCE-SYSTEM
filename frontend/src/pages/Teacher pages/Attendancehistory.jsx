import React, { useEffect, useState } from "react";
import { Filter, Eye, X, Calendar } from "lucide-react";
import api from "../../api/axious";
import "./AttendanceHistory.css";

const SUBJECT_OPTIONS = [
  "All Subjects",
  "Machine Learning",
  "DBMS",
  "Computer Networks",
];

const CLASS_OPTIONS = [
  "TY CSD - D",
  "TY CSD - C",
  "TY CSD - B",
  "TY CSD - A",
];

function SessionDetailsModal({ session, onClose }) {
  if (!session) return null;

  return (
    <div className="ah-modal-overlay" onClick={onClose}>
      <div className="ah-modal" onClick={(e) => e.stopPropagation()}>
        <div className="ah-modal__header">
          <h3>Session Details</h3>

          <button
            type="button"
            className="ah-modal__close"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        <div className="ah-modal__grid">
          <div>
            <span>Date</span>
            <strong>{session.date}</strong>
          </div>

          <div>
            <span>Subject</span>
            <strong>{session.subject}</strong>
          </div>

          <div>
            <span>Class</span>
            <strong>{session.classDivision}</strong>
          </div>

          <div>
            <span>Status</span>
            <strong>{session.status}</strong>
          </div>

          <div>
            <span>Present</span>
            <strong className="ah-present">{session.present}</strong>
          </div>

          <div>
            <span>Absent</span>
            <strong className="ah-absent">{session.absent}</strong>
          </div>

          <div>
            <span>Attendance</span>
            <strong>{session.attendance}</strong>
          </div>
        </div>

        <p className="ah-modal__note">
          Detailed per-student breakdown for this session will appear here
          once connected to the attendance records API.
        </p>
      </div>
    </div>
  );
}

export default function AttendanceHistory() {
  const [dateFilter, setDateFilter] = useState("");
  const [subjectFilter, setSubjectFilter] = useState("All Subjects");
  const [classFilter, setClassFilter] = useState("TY CSD - D");

  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedSession, setSelectedSession] = useState(null);

  // ---------------------------------------------------------
  // Fetch attendance history
  // ---------------------------------------------------------

  const fetchAttendanceHistory = async (filters = {}) => {
    try {
      setLoading(true);
      setError("");

      const params = {};

      if (filters.date) {
        params.date = filters.date;
      }

      if (
        filters.subject &&
        filters.subject !== "All Subjects"
      ) {
        params.subject = filters.subject;
      }

      if (filters.classDivision) {
        params.classDivision = filters.classDivision;
      }

      const response = await api.get(
        "/attendance/teacher/history",
        {
          params,
        }
      );

      setHistory(response.data.history || []);
    } catch (err) {
      console.error(
        "Failed to fetch attendance history:",
        err
      );

      setError(
        err.response?.data?.message ||
          "Failed to load attendance history."
      );

      setHistory([]);
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------------------------
  // Load history when page opens
  // ---------------------------------------------------------

  useEffect(() => {
    fetchAttendanceHistory();
  }, []);

  // ---------------------------------------------------------
  // Apply filters
  // ---------------------------------------------------------

  const handleApply = () => {
    fetchAttendanceHistory({
      date: dateFilter,
      subject: subjectFilter,
      classDivision: classFilter,
    });
  };

  return (
    <div className="ah-page">
      <div className="ah-header">
        <h1>Attendance History</h1>
        <p>
          View previous attendance sessions and class records.
        </p>
      </div>

      {/* Filters */}
      <div className="ah-card ah-filters">
        <label className="ah-field">
          <span>Date</span>

          <div className="ah-field__control">
            <Calendar size={16} />

            <input
              type="date"
              value={dateFilter}
              onChange={(e) =>
                setDateFilter(e.target.value)
              }
            />
          </div>
        </label>

        <label className="ah-field">
          <span>Subject</span>

          <select
            value={subjectFilter}
            onChange={(e) =>
              setSubjectFilter(e.target.value)
            }
          >
            {SUBJECT_OPTIONS.map((subject) => (
              <option key={subject} value={subject}>
                {subject}
              </option>
            ))}
          </select>
        </label>

        <label className="ah-field">
          <span>Class</span>

          <select
            value={classFilter}
            onChange={(e) =>
              setClassFilter(e.target.value)
            }
          >
            {CLASS_OPTIONS.map((className) => (
              <option key={className} value={className}>
                {className}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          className="ah-btn ah-btn--green"
          onClick={handleApply}
          disabled={loading}
        >
          <Filter size={16} />

          {loading ? "Loading..." : "Apply Filters"}
        </button>
      </div>

      {/* Table */}
      <div className="ah-card">
        {loading ? (
          <p className="ah-empty">
            Loading attendance history...
          </p>
        ) : error ? (
          <p className="ah-empty">
            {error}
          </p>
        ) : (
          <>
            <table className="ah-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Subject</th>
                  <th>Class</th>
                  <th>Present</th>
                  <th>Absent</th>
                  <th>Attendance</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>

              <tbody>
                {history.map((row) => (
                  <tr key={row.id}>
                    <td>{row.date}</td>

                    <td>{row.subject}</td>

                    <td>{row.classDivision}</td>

                    <td className="ah-present">
                      {row.present}
                    </td>

                    <td className="ah-absent">
                      {row.absent}
                    </td>

                    <td>{row.attendance}</td>

                    <td>
                      <span className="ah-badge">
                        {row.status}
                      </span>
                    </td>

                    <td>
                      <button
                        type="button"
                        className="ah-btn ah-btn--outline ah-btn--sm"
                        onClick={() =>
                          setSelectedSession(row)
                        }
                      >
                        <Eye size={14} />
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {history.length === 0 && (
              <p className="ah-empty">
                No attendance sessions found.
              </p>
            )}

            {/* Mobile cards */}
            <div className="ah-table-cards">
              {history.map((row) => (
                <div
                  className="ah-table-card"
                  key={row.id}
                >
                  <div className="ah-table-card__head">
                    <strong>{row.subject}</strong>

                    <span className="ah-badge">
                      {row.status}
                    </span>
                  </div>

                  <p className="ah-table-card__meta">
                    {row.classDivision} &bull; {row.date}
                  </p>

                  <div className="ah-table-card__stats">
                    <span className="ah-present">
                      Present: {row.present}
                    </span>

                    <span className="ah-absent">
                      Absent: {row.absent}
                    </span>

                    <span>
                      {row.attendance}
                    </span>
                  </div>

                  <button
                    type="button"
                    className="ah-btn ah-btn--outline ah-btn--sm ah-table-card__btn"
                    onClick={() =>
                      setSelectedSession(row)
                    }
                  >
                    <Eye size={14} />
                    View
                  </button>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Details Modal */}
      <SessionDetailsModal
        session={selectedSession}
        onClose={() => setSelectedSession(null)}
      />
    </div>
  );
}