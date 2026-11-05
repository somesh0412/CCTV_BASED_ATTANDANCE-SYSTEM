import React, { useState } from "react";
import { Search, Eye, X, CheckCircle2, AlertTriangle } from "lucide-react";
import "./Students.css";

/* Mock data — swap for GET /api/teacher/students later */
const STUDENTS = [
  { id: "CSD001", name: "Rahul Sharma", email: "rahul@college.edu", faceStatus: "registered", attendance: "93%", overall: "93%", subjectWise: [{ subject: "Machine Learning", value: "90%" }, { subject: "DBMS", value: "95%" }, { subject: "Computer Networks", value: "94%" }] },
  { id: "CSD002", name: "Priya Patil", email: "priya@college.edu", faceStatus: "registered", attendance: "96%", overall: "96%", subjectWise: [{ subject: "Machine Learning", value: "98%" }, { subject: "DBMS", value: "95%" }, { subject: "Computer Networks", value: "95%" }] },
  { id: "CSD003", name: "Amit Shah", email: "amit@college.edu", faceStatus: "not_registered", attendance: "82%", overall: "82%", subjectWise: [{ subject: "Machine Learning", value: "78%" }, { subject: "DBMS", value: "85%" }, { subject: "Computer Networks", value: "83%" }] },
  { id: "CSD004", name: "Sneha Kulkarni", email: "sneha@college.edu", faceStatus: "registered", attendance: "97%", overall: "97%", subjectWise: [{ subject: "Machine Learning", value: "96%" }, { subject: "DBMS", value: "99%" }, { subject: "Computer Networks", value: "96%" }] },
  { id: "CSD005", name: "Rohan Deshmukh", faceStatus: "not_registered", email: "rohan@college.edu", attendance: "74%", overall: "74%", subjectWise: [{ subject: "Machine Learning", value: "70%" }, { subject: "DBMS", value: "76%" }, { subject: "Computer Networks", value: "76%" }] },
];

const CLASS_OPTIONS = ["TY CSD - D", "TY CSD - C", "TY CSD - B", "TY CSD - A"];

function StudentDetailsModal({ student, onClose }) {
  if (!student) return null;
  const registered = student.faceStatus === "registered";
  return (
    <div className="st-modal-overlay" onClick={onClose}>
      <div className="st-modal" onClick={(e) => e.stopPropagation()}>
        <div className="st-modal__header">
          <h3>Student Details</h3>
          <button type="button" className="st-modal__close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div className="st-modal__profile">
          <span className="st-avatar">{student.name.split(" ").map((w) => w[0]).slice(0, 2).join("")}</span>
          <div>
            <strong>{student.name}</strong>
            <p>{student.id}</p>
          </div>
        </div>

        <div className="st-modal__grid">
          <div><span>Class</span><strong>TY CSD - D</strong></div>
          <div><span>Email</span><strong>{student.email}</strong></div>
          <div>
            <span>Face Registration</span>
            <strong className={registered ? "st-registered" : "st-not-registered"}>
              {registered ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
              {registered ? "Registered" : "Not Registered"}
            </strong>
          </div>
          <div><span>Overall Attendance</span><strong>{student.overall}</strong></div>
        </div>

        <h4 className="st-modal__subhead">Subject-wise Attendance</h4>
        <div className="st-subject-list">
          {student.subjectWise.map((s) => (
            <div className="st-subject-row" key={s.subject}>
              <span>{s.subject}</span>
              <div className="st-progress">
                <div className="st-progress__fill" style={{ width: s.value }} />
              </div>
              <strong>{s.value}</strong>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Students() {
  const [classFilter, setClassFilter] = useState("TY CSD - D");
  const [search, setSearch] = useState("");
  const [selectedStudent, setSelectedStudent] = useState(null);

  const filtered = STUDENTS.filter((s) =>
    s.name.toLowerCase().includes(search.toLowerCase()) ||
    s.id.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="st-page">
      <div className="st-header">
        <h1>Students</h1>
        <p>View students and their attendance information.</p>
      </div>

      <div className="st-card">
        <div className="st-toolbar">
          <label className="st-class-select">
            <span>Class</span>
            <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
              {CLASS_OPTIONS.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <div className="st-search">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search student..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>

        <table className="st-table">
          <thead>
            <tr>
              <th>Student ID</th>
              <th>Name</th>
              <th>Email</th>
              <th>Face Status</th>
              <th>Attendance</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((s) => (
              <tr key={s.id}>
                <td>{s.id}</td>
                <td>{s.name}</td>
                <td>{s.email}</td>
                <td>
                  <span className={s.faceStatus === "registered" ? "st-registered" : "st-not-registered"}>
                    {s.faceStatus === "registered" ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                    {s.faceStatus === "registered" ? "Registered" : "Not Registered"}
                  </span>
                </td>
                <td>{s.attendance}</td>
                <td>
                  <button type="button" className="st-btn st-btn--outline st-btn--sm" onClick={() => setSelectedStudent(s)}>
                    <Eye size={14} /> View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filtered.length === 0 && <p className="st-empty">No students match your search.</p>}

        <div className="st-table-cards">
          {filtered.map((s) => (
            <div className="st-table-card" key={s.id}>
              <div className="st-table-card__head">
                <strong>{s.name}</strong>
                <span className={s.faceStatus === "registered" ? "st-registered" : "st-not-registered"}>
                  {s.faceStatus === "registered" ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                  {s.faceStatus === "registered" ? "Registered" : "Not Registered"}
                </span>
              </div>
              <p className="st-table-card__meta">{s.id} &bull; {s.email}</p>
              <p className="st-table-card__meta">Attendance: {s.attendance}</p>
              <button type="button" className="st-btn st-btn--outline st-btn--sm st-table-card__btn" onClick={() => setSelectedStudent(s)}>
                <Eye size={14} /> View
              </button>
            </div>
          ))}
        </div>
      </div>

      <p className="st-note">
        Students register their own face through the Student Portal — this page only
        reflects their registration status.
      </p>

      <StudentDetailsModal student={selectedStudent} onClose={() => setSelectedStudent(null)} />
    </div>
  );
}