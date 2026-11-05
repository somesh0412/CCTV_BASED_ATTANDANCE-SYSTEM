import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  Bell,
  BookOpen,
  Calendar,
  CheckCircle,
  ChevronDown,
  Clock,
  LineChart,
  LogOut,
  Menu,
  User,
  X,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import api from "../../api/axious";
import PageLogo from "../../assets/cctvLogo.png";
import "./StudentDashboard.css";

const EMPTY_DASHBOARD = {
  student: null,
  attendance: {
    total: 0,
    attended: 0,
    missed: 0,
    percentage: null,
    subjects: [],
    records: [],
  },
  schedules: [],
};

function getErrorMessage(error) {
  return error.response?.data?.message || "Unable to load your dashboard. Please try again.";
}

function formatDate(value) {
  if (!value) return "Unknown date";
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatToday() {
  return new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function EmptyState({ children }) {
  return <div className="empty-state">{children}</div>;
}

function SummaryCard({ icon: Icon, title, value, isPercentage = false }) {
  return (
    <div className="summary-card">
      <div className="card-header">
        <h3>{title}</h3>
        <Icon size={24} className="card-icon" />
      </div>
      <div className="card-value">
        {value ?? "—"}
        {isPercentage && value !== null && value !== undefined ? "%" : ""}
      </div>
    </div>
  );
}

function ScheduleCard({ schedule }) {
  const status = schedule.status || "upcoming";
  const statusLabel = status.charAt(0).toUpperCase() + status.slice(1);

  return (
    <div className={`schedule-card status-${status === "completed" ? "present" : "upcoming"}`}>
      <div className="schedule-time">{schedule.time || "Time not available"}</div>
      <div className="schedule-details">
        <h4>{schedule.subject}</h4>
        <p className="schedule-class">{schedule.classDivision}</p>
        <p className="schedule-room">Room {schedule.room}</p>
      </div>
      <div className={`schedule-status ${status === "completed" ? "present" : "upcoming"}`}>
        {status === "completed" ? <CheckCircle size={18} /> : <Clock size={18} />}
        {statusLabel}
      </div>
    </div>
  );
}

function AttendanceTable({ records }) {
  return (
    <div className="attendance-section">
      <h2>Recent Attendance</h2>
      {!records.length ? (
        <EmptyState>No attendance records yet.</EmptyState>
      ) : (
        <div className="table-wrapper">
          <table className="attendance-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Subject</th>
                <th>Class</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {records.map((record, index) => (
                <tr key={`${record.date}-${record.subject}-${index}`}>
                  <td>{formatDate(record.date)}</td>
                  <td>{record.subject}</td>
                  <td>{record.classDivision}</td>
                  <td>
                    <span className={`status-badge ${record.status.toLowerCase()}`}>
                      <span className="status-dot" />
                      {record.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function SubjectAttendance({ subjects }) {
  return (
    <div className="subject-attendance-section">
      <h2>Subject-wise Attendance</h2>
      {!subjects.length ? (
        <EmptyState>No subject attendance records yet.</EmptyState>
      ) : (
        <div className="subjects-grid">
          {subjects.map((subject) => (
            <div key={subject.name} className="subject-card">
              <div className="subject-header">
                <h4>{subject.name}</h4>
                <span className="subject-percentage">
                  {subject.percentage === null ? "—" : `${subject.percentage}%`}
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${subject.percentage || 0}%` }}
                />
              </div>
              <small>{subject.attended} of {subject.total} classes attended</small>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function FaceRecognitionStatus({ student }) {
  const registered = student?.faceRegistered === true;

  return (
    <div className="face-recognition-card">
      <div className="face-card-header">
        <h3>Face Recognition Profile</h3>
      </div>
      <div className="face-card-content">
        <div className="status-item">
          <span className="status-label">Status:</span>
          <div className="status-value">
            {registered ? <CheckCircle size={18} className="success-icon" /> : <AlertCircle size={18} />}
            <span>{registered ? "Registered" : "Not registered"}</span>
          </div>
        </div>
        <div className="status-item">
          <span className="status-label">Last Updated:</span>
          <span className="status-value-text">
            {student?.faceUpdatedAt ? formatDate(student.faceUpdatedAt.split("T")[0]) : "Not available"}
          </span>
        </div>
      </div>
    </div>
  );
}

function ProfilePage({ student }) {
  const fullName = `${student.firstName} ${student.lastName}`.trim();
  const details = [
    ["Name", fullName],
    ["Student ID", student.studentId],
    ["Department", student.department],
    ["Class / Division", student.classDivision],
    ["Email", student.email],
  ];

  return (
    <div className="profile-page">
      <div className="profile-header">
        <div className="profile-avatar-large">{student.firstName.charAt(0).toUpperCase()}</div>
        <h1>{fullName}</h1>
      </div>
      <div className="profile-info-grid">
        {details.map(([label, value]) => (
          <div className="profile-info-item" key={label}>
            <label>{label}</label>
            <p>{value || "Not available"}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function Sidebar({ activeMenu, setActiveMenu, sidebarOpen, setSidebarOpen, onLogout }) {
  const menuItems = [
    { id: "dashboard", label: "Dashboard", icon: BarChart3 },
    { id: "schedule", label: "My Schedule", icon: Calendar },
    { id: "attendance", label: "My Attendance", icon: BookOpen },
    { id: "report", label: "Attendance Report", icon: LineChart },
    { id: "profile", label: "Profile", icon: User },
  ];

  return (
    <>
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <div className="logo">
            <img src={PageLogo} alt="" className="sidebar-logo-image" />
            <span>CCTV ATTENDANCE SYSTEM</span>
          </div>
          <button className="close-btn" onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <X size={24} />
          </button>
        </div>
        <nav className="sidebar-nav">
          {menuItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${activeMenu === id ? "active" : ""}`}
              onClick={() => {
                setActiveMenu(id);
                setSidebarOpen(false);
              }}
            >
              <Icon size={20} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <button className="nav-item logout-btn" onClick={onLogout}>
          <LogOut size={20} />
          <span>Logout</span>
        </button>
      </aside>
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}
    </>
  );
}

function Header({ student, sidebarOpen, setSidebarOpen }) {
  const fullName = `${student.firstName} ${student.lastName}`.trim();

  return (
    <header className="header">
      <div className="header-left">
        <button className="hamburger-btn" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="Open menu">
          <Menu size={24} />
        </button>
        <div className="welcome-text">
          <h1>Welcome, {student.firstName} 👋</h1>
          <p>Here&apos;s your attendance overview.</p>
        </div>
      </div>
      <div className="header-right">
        <button className="notification-btn" aria-label="Notifications"><Bell size={20} /></button>
        <div className="profile-btn">
          <div className="profile-avatar">{student.firstName.charAt(0).toUpperCase()}</div>
          <div className="profile-info"><span className="profile-name">{fullName}</span></div>
          <ChevronDown size={18} />
        </div>
      </div>
    </header>
  );
}

function DashboardContent({ activeMenu, dashboard }) {
  const { student, attendance, schedules } = dashboard;
  const today = formatToday();

  if (activeMenu === "profile") return <ProfilePage student={student} />;
  if (activeMenu === "schedule") {
    return (
      <div className="content">
        <h2>My Schedule</h2>
        <p className="section-date">{today}</p>
        {!schedules.length ? <EmptyState>No schedule available for today.</EmptyState> : (
          <div className="schedule-cards">{schedules.map((schedule) => <ScheduleCard key={schedule.id} schedule={schedule} />)}</div>
        )}
      </div>
    );
  }
  if (activeMenu === "attendance") {
    return (
      <div className="content">
        <h2>My Attendance</h2>
        <div className="attendance-overview">
          <SummaryCard icon={BookOpen} title="Total Classes" value={attendance.total || 0} />
          <SummaryCard icon={CheckCircle} title="Classes Attended" value={attendance.attended || 0} />
          <SummaryCard icon={AlertCircle} title="Classes Missed" value={attendance.missed || 0} />
          <SummaryCard icon={BarChart3} title="Overall Attendance" value={attendance.percentage} isPercentage />
        </div>
        <SubjectAttendance subjects={attendance.subjects} />
      </div>
    );
  }
  if (activeMenu === "report") {
    return <div className="content"><h2>Attendance Report</h2><AttendanceTable records={attendance.records} /></div>;
  }

  return (
    <div className="content">
      <div className="attendance-summary">
        <SummaryCard icon={BookOpen} title="Total Classes" value={attendance.total || 0} />
        <SummaryCard icon={CheckCircle} title="Classes Attended" value={attendance.attended || 0} />
        <SummaryCard icon={AlertCircle} title="Classes Missed" value={attendance.missed || 0} />
        <SummaryCard icon={BarChart3} title="Overall Attendance" value={attendance.percentage} isPercentage />
      </div>
      <section className="today-schedule-section">
        <div className="section-header"><h2>Today&apos;s Schedule</h2><p className="section-date">{today}</p></div>
        {!schedules.length ? <EmptyState>No schedule available for today.</EmptyState> : (
          <div className="schedule-cards">{schedules.map((schedule) => <ScheduleCard key={schedule.id} schedule={schedule} />)}</div>
        )}
      </section>
      <div className="dashboard-grid">
        <SubjectAttendance subjects={attendance.subjects} />
        <FaceRecognitionStatus student={student} />
      </div>
      <AttendanceTable records={attendance.records} />
    </div>
  );
}

export default function StudentDashboard() {
  const navigate = useNavigate();
  const [activeMenu, setActiveMenu] = useState("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dashboard, setDashboard] = useState(EMPTY_DASHBOARD);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadDashboard() {
      try {
        const response = await api.get("/auth/student/dashboard");
        if (active) setDashboard(response.data.data);
      } catch (requestError) {
        if (active) setError(getErrorMessage(requestError));
      } finally {
        if (active) setLoading(false);
      }
    }

    loadDashboard();
    return () => { active = false; };
  }, []);

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("student");
    navigate("/student-login", { replace: true });
  };

  const content = useMemo(() => {
    if (loading) return <div className="dashboard-state" role="status">Loading your dashboard...</div>;
    if (error) return <div className="dashboard-state dashboard-error" role="alert">{error}</div>;
    if (!dashboard.student) return <div className="dashboard-state dashboard-error">Student profile is unavailable.</div>;
    return <DashboardContent activeMenu={activeMenu} dashboard={dashboard} />;
  }, [activeMenu, dashboard, error, loading]);

  return (
    <div className="student-dashboard">
      {dashboard.student && (
        <Sidebar
          activeMenu={activeMenu}
          setActiveMenu={setActiveMenu}
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          onLogout={logout}
        />
      )}
      <div className="main-container">
        {dashboard.student && <Header student={dashboard.student} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />}
        <main className="main-content">{content}</main>
      </div>
    </div>
  );
}
