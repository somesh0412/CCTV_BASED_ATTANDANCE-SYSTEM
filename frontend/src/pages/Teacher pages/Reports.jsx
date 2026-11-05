import React, { useEffect, useState } from "react";
import { Users, TrendingUp, Award, AlertTriangle, FileBarChart, Download, CheckCircle2 } from "lucide-react";
import "./Reports.css";
import api from "../../api/axious";

/* Mock data — swap for GET /api/teacher/reports later */

const CLASS_OPTIONS = ["TY CSD - D", "TY CSD - C", "TY CSD - B", "TY CSD - A"];
const SUBJECT_OPTIONS = ["All Subjects", "Machine Learning", "DBMS", "Computer Networks", "Data Science"];

export default function Reports() {
  const [classFilter, setClassFilter] = useState("TY CSD - D");
  const [subjectFilter, setSubjectFilter] = useState("All Subjects");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [reportMessage, setReportMessage] = useState("");
  const [exportMessage, setExportMessage] = useState("");
  const [reportData, setReportData] = useState({
  summary: {
    totalStudents: 0,
    averageAttendance: 0,
    highestAttendance: 0,
    lowAttendanceCount: 0,
  },
  subjectWise: [],
  lowAttendance: [],
});

const [loading, setLoading] = useState(false);
const [error, setError] = useState("");
const fetchReports = async () => {
  try {
    setLoading(true);
    setError("");

    const params = {};

    if (classFilter) {
      params.classDivision = classFilter;
    }

    if (
      subjectFilter &&
      subjectFilter !== "All Subjects"
    ) {
      params.subject = subjectFilter;
    }

    if (startDate) {
      params.startDate = startDate;
    }

    if (endDate) {
      params.endDate = endDate;
    }

    const response = await api.get(
      "/attendance/teacher/reports",
      {
        params,
      }
    );

    setReportData(response.data);

  } catch (error) {
    console.error(
      "Failed to load reports:",
      error
    );

    setError(
      error.response?.data?.message ||
        "Failed to load attendance reports."
    );
  } finally {
    setLoading(false);
  }
};
useEffect(() => {
  fetchReports();
}, []);

  const handleGenerateReport = async () => {
  await fetchReports();

    setReportMessage(
      "Report generated successfully."
    );

    setTimeout(
      () => setReportMessage(""),
      3000
    );
  };

  const handleExportCsv = async () => {
  try {
    setExportMessage("");

    const params = {};

    if (classFilter) {
      params.classDivision = classFilter;
    }

    if (
      subjectFilter &&
      subjectFilter !== "All Subjects"
    ) {
      params.subject = subjectFilter;
    }

    if (startDate) {
      params.startDate = startDate;
    }

    if (endDate) {
      params.endDate = endDate;
    }

    const response = await api.get(
      "/attendance/teacher/reports/export",
      {
        params,
        responseType: "blob",
      }
    );

    // Create downloadable file
    const blob = new Blob(
      [response.data],
      {
        type: "text/csv;charset=utf-8;",
      }
    );

    const url =
      window.URL.createObjectURL(blob);

    const link =
      document.createElement("a");

    link.href = url;
    link.download =
      "attendance-report.csv";

    document.body.appendChild(link);

    link.click();

    link.remove();

    window.URL.revokeObjectURL(url);

    setExportMessage(
      "CSV exported successfully."
    );

    setTimeout(
      () => setExportMessage(""),
      3000
    );

  } catch (error) {
    console.error(
      "CSV export failed:",
      error
    );

    setExportMessage(
      error.response?.data?.message ||
        "Failed to export CSV."
    );
  }
};
  return (
    <div className="rp-page">
      <div className="rp-header">
        <h1>Attendance Reports</h1>
        <p>Analyze attendance performance and generate reports.</p>
      </div>

      <div className="rp-card rp-filters">
        <label className="rp-field">
          <span>Class</span>
          <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
            {CLASS_OPTIONS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>

        <label className="rp-field">
          <span>Subject</span>
          <select value={subjectFilter} onChange={(e) => setSubjectFilter(e.target.value)}>
            {SUBJECT_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>

        <label className="rp-field">
          <span>Start Date</span>
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </label>

        <label className="rp-field">
          <span>End Date</span>
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
        </label>
      </div>

      <div className="rp-summary">
  {[
    {
      key: "total",
      label: "Total Students",
      value: reportData.summary.totalStudents,
      icon: Users,
      tone: "blue",
    },
    {
      key: "avg",
      label: "Average Attendance",
      value: `${reportData.summary.averageAttendance}%`,
      icon: TrendingUp,
      tone: "green",
    },
    {
      key: "high",
      label: "Highest Attendance",
      value: `${reportData.summary.highestAttendance}%`,
      icon: Award,
      tone: "purple",
    },
    {
      key: "low",
      label: "Low Attendance",
      value: `${reportData.summary.lowAttendanceCount} Students`,
      icon: AlertTriangle,
      tone: "red",
    },
  ].map(
    ({
      key,
      label,
      value,
      icon: Icon,
      tone,
    }) => (
      <div
        className="rp-summary-card"
        key={key}
      >
        <span
          className={`rp-summary-card__icon rp-summary-card__icon--${tone}`}
        >
          <Icon size={20} />
        </span>

        <div>
          <p>{label}</p>
          <strong>{value}</strong>
        </div>
      </div>
    )
  )}
</div>
              

      <div className="rp-grid">
        <div className="rp-card">
          <h2>Subject-wise Attendance</h2>
          <div className="rp-bars">
            {reportData.subjectWise.map((s) => (
              <div className="rp-bar-row" key={s.subject}>
                <span className="rp-bar-row__label">{s.subject}</span>
                <div className="rp-bar-track">
                  <div
                    className={`rp-bar-fill ${s.value < 75 ? "rp-bar-fill--low" : ""}`}
                    style={{ width: `${s.value}%` }}
                  />
                </div>
                <span className="rp-bar-row__value">{s.value}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rp-card">
          <h2>Low Attendance Students</h2>
          <div className="rp-low-list">
            {reportData.lowAttendance.map((s) => (
              <div className="rp-low-row" key={s.studentId}>
                <div>
                  <strong>{s.name}</strong>
                  <p>{s.attendance} attendance</p>
                </div>
                <span className="rp-badge-low">
                  <AlertTriangle size={12} /> {s.status}
                </span>
              </div>
            ))}
            {reportData.lowAttendance.length === 0 && (
              <p className="rp-empty">No students below the attendance threshold.</p>
            )}
          </div>
        </div>
      </div>

      <div className="rp-card rp-actions">
        <div className="rp-actions__buttons">
          <button type="button" className="rp-btn rp-btn--green" onClick={handleGenerateReport}>
            <FileBarChart size={16} /> Generate Report
          </button>
          <button type="button" className="rp-btn rp-btn--outline" onClick={handleExportCsv}>
            <Download size={16} /> Export CSV
          </button>
        </div>

        {reportMessage && (
          <p className="rp-message rp-message--success">
            <CheckCircle2 size={15} /> {reportMessage}
          </p>
        )}
        {exportMessage && <p className="rp-message">{exportMessage}</p>}
      </div>
    </div>
  );
}