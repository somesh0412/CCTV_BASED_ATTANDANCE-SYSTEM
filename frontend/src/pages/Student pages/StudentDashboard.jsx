import { useEffect, useState } from "react";
import {
  ShieldCheck,
  Menu,
  X,
  Bell,
  ChevronDown,
  LayoutDashboard,
  CalendarClock,
  History,
  FileBarChart,
  Settings,
  LogOut,
  Clock,
  MapPin,
  Eye,
  GraduationCap,
  UserCheck,
  UserX,
  TrendingUp,
  User
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import api from "../../api/axious";
import "./StudentDashboard.css";

/* ----------------------------------------------------------
   Static / mock data
---------------------------------------------------------- */

const MENU_ITEMS = [
  {
    key: "dashboard",
    label: "Dashboard",
    icon: LayoutDashboard,
  },
  {
    key: "schedule",
    label: "My Schedule",
    icon: CalendarClock,
  },
  {
    key: "history",
    label: "My Attendance",
    icon: History,
  },
  {
    key: "reports",
    label: "Reports",
    icon: FileBarChart,
  },
  {
    key: "profile",
    label: "Profile",
    icon: Settings,
  },
];

const OVERVIEW_CARDS = [
  {
    key: "total",
    label: "Total Classes",
    value: 42,
    icon: GraduationCap,
    tone: "blue",
  },
  {
    key: "present",
    label: "Classes Attended",
    value: 38,
    icon: UserCheck,
    tone: "green",
  },
  {
    key: "absent",
    label: "Classes Missed",
    value: 4,
    icon: UserX,
    tone: "red",
  },
  {
    key: "rate",
    label: "Overall Rate",
    value: "90.5%",
    icon: TrendingUp,
    tone: "purple",
  },
];

const INITIAL_SCHEDULE = [
  {
    id: "s1",
    startTime: "10:15 AM",
    endTime: "11:15 AM",
    subject: "Machine Learning",
    classDivision: "TY CSD - D",
    room: "Room 201",
    status: "live",
  },
  {
    id: "s2",
    startTime: "12:00 PM",
    endTime: "1:00 PM",
    subject: "Database Management System",
    classDivision: "TY CSD - D",
    room: "Room 304",
    status: "upcoming",
  },
  {
    id: "s3",
    startTime: "2:00 PM",
    endTime: "3:00 PM",
    subject: "Computer Networks",
    classDivision: "TY CSD - D",
    room: "Room 205",
    status: "upcoming",
  },
];

const TODAY_LABEL = new Date().toLocaleDateString("en-US", {
  weekday: "long",
  day: "numeric",
  month: "long",
  year: "numeric",
});

/* ----------------------------------------------------------
   Overview Card
---------------------------------------------------------- */
function OverviewCard({ icon: Icon, label, value, tone }) {
  return (
    <div className="sd-overview-card">
      <span className={`sd-overview-card__icon sd-overview-card__icon--${tone}`}>
        <Icon size={20} />
      </span>
      <div>
        <p className="sd-overview-card__label">{label}</p>
        <strong className="sd-overview-card__value">{value}</strong>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------
   Schedule Item
---------------------------------------------------------- */
function ScheduleItem({ item }) {
  const isLive = item.status === "live";

  return (
    <div className={`sd-schedule-item ${isLive ? "sd-schedule-item--live" : ""}`}>
      <div className="sd-schedule-item__time">
        <Clock size={16} />
        <span>
          {item.startTime} – {item.endTime}
        </span>
      </div>

      <div className="sd-schedule-item__info">
        <h4>{item.subject}</h4>
        <p>
          {item.classDivision}
          <span className="sd-dot">&bull;</span>
          <MapPin size={13} />
          {item.room}
        </p>
      </div>

      <div className="sd-schedule-item__status">
        <span className={`sd-badge ${isLive ? "sd-badge--live" : "sd-badge--upcoming"}`}>
          {isLive ? "Class Running" : "Upcoming"}
        </span>
      </div>

      <div className="sd-schedule-item__action">
        {isLive ? (
          <button type="button" className="sd-btn sd-btn--blue">
            <Eye size={16} /> Check In Status
          </button>
        ) : (
          <button type="button" className="sd-btn sd-btn--outline">
            <Eye size={16} /> View Details
          </button>
        )}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------
   Recent Attendance
---------------------------------------------------------- */
function RecentAttendanceTable({ rows }) {
  return (
    <table className="sd-table">
      <thead>
        <tr>
          <th>Subject</th>
          <th>Class</th>
          <th>Date</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id}>
            <td>{row.subject}</td>
            <td>{row.classDivision}</td>
            <td>{row.date}</td>
            <td className={row.status === "Present" ? "sd-table__present" : "sd-table__absent"}>
              {row.status}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ProfileContent({ studentData }) {
  const displayName = studentData.name || `${studentData.firstName || ""} ${studentData.lastName || ""}`.trim() || "Student";
  const initials = displayName
    .split(/\s+/)
    .map((word) => word[0])
    .slice(0, 2)
    .join("");

  return (
    <div className="sd-card sd-profile-card">
      <div className="sd-profile-card__avatar">
        {initials}
      </div>
      <h2>{displayName}</h2>
      <p style={{ color: "var(--sd-text-muted)", marginBottom: "20px" }}>{studentData.email}</p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', textAlign: 'left', maxWidth: '400px', margin: '0 auto' }}>
        <div>
          <label style={{ fontSize: '0.8rem', color: 'var(--sd-text-muted)', fontWeight: 600 }}>Student ID</label>
          <p style={{ margin: '4px 0 0 0', fontWeight: 500 }}>{studentData.studentId}</p>
        </div>
        <div>
          <label style={{ fontSize: '0.8rem', color: 'var(--sd-text-muted)', fontWeight: 600 }}>Department</label>
          <p style={{ margin: '4px 0 0 0', fontWeight: 500 }}>{studentData.department}</p>
        </div>
        <div>
          <label style={{ fontSize: '0.8rem', color: 'var(--sd-text-muted)', fontWeight: 600 }}>Face Registration</label>
          <p style={{ margin: '4px 0 0 0', fontWeight: 500, color: 'var(--sd-accent-green)' }}>Verified Status</p>
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------
   MAIN COMPONENT
---------------------------------------------------------- */
export default function StudentDashboard() {
  const navigate = useNavigate();
  // Fetch from localStorage if available
  const storedStudent = JSON.parse(localStorage.getItem('student'));
  const studentData = storedStudent || {
    name: "Somesh Raut",
    email: "student@college.edu",
    studentId: "CSD-D-28",
    department: "Computer Science and Design"
  };
  const displayName = studentData.name || `${studentData.firstName || ""} ${studentData.lastName || ""}`.trim() || "Student";
  const initials = displayName
    .split(/\s+/)
    .map((word) => word[0])
    .slice(0, 2)
    .join("");

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeMenu, setActiveMenu] = useState("dashboard");
  const [profileOpen, setProfileOpen] = useState(false);
  const [schedule] = useState(INITIAL_SCHEDULE);
  const [attendance, setAttendance] = useState([]);

  useEffect(() => {
    if (!localStorage.getItem("token") || !storedStudent) {
      navigate("/student-login", { replace: true });
    }
  }, [navigate, storedStudent]);

  useEffect(() => {
    let active = true;

    api.get("/attendance/mine")
      .then((response) => {
        if (active) setAttendance(response.data.attendance || []);
      })
      .catch(() => {
        if (active) setAttendance([]);
      });

    return () => {
      active = false;
    };
  }, []);

  const presentCount = attendance.filter((record) => record.status === "Present").length;
  const absentCount = attendance.filter((record) => record.status === "Absent").length;
  const totalCount = attendance.length;
  const attendanceRate = totalCount ? `${Math.round((presentCount / totalCount) * 100)}%` : "0%";
  const overviewCards = OVERVIEW_CARDS.map((card) => ({
    ...card,
    value: {
      total: totalCount,
      present: presentCount,
      absent: absentCount,
      rate: attendanceRate,
    }[card.key],
  }));
  const attendanceRows = attendance.map((record) => ({
    id: record._id,
    subject: "Recognized attendance",
    classDivision: studentData.department,
    date: record.date,
    status: record.status,
  }));

  const handleMenuClick = (key) => {
    setActiveMenu(key);
    setSidebarOpen(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("student");
    navigate("/");
  };

  return (
    <div className="sd-page">
      {/* ====================================================
          HEADER
      ==================================================== */}
      <header className="sd-header">
        <div className="sd-header__left">
          <button
            type="button"
            className="sd-hamburger"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
          >
            <Menu size={22} />
          </button>
          <div className="sd-header__brand">
            <span className="sd-header__brand-icon">
              <ShieldCheck size={20} />
            </span>
            <span className="sd-header__brand-text">
              CCTV ATTENDANCE SYSTEM
            </span>
          </div>
        </div>

        <div className="sd-header__right">
          <button type="button" className="sd-icon-btn" aria-label="Notifications">
            <Bell size={19} />
            <span className="sd-icon-btn__dot" />
          </button>

          <div className="sd-profile">
            <button
              type="button"
              className="sd-profile__trigger"
              onClick={() => setProfileOpen((v) => !v)}
            >
              <span className="sd-profile__avatar">
                {initials}
              </span>
              <span className="sd-profile__name">{displayName}</span>
              <ChevronDown size={16} />
            </button>

            {profileOpen && (
              <div className="sd-profile__dropdown">
                <button type="button" onClick={() => handleMenuClick("profile")}>
                  <User size={15} /> My Profile
                </button>
                <button type="button" onClick={handleLogout} className="sd-profile__logout">
                  <LogOut size={15} /> Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="sd-body">
        {/* ==================================================
            DESKTOP SIDEBAR
        ================================================== */}
        <aside className="sd-sidebar sd-sidebar--desktop">
          <nav className="sd-sidebar__nav">
            {MENU_ITEMS.map(({ key, label, icon: Icon }) => (
              <button
                type="button"
                key={key}
                className={`sd-sidebar__item ${activeMenu === key ? "sd-sidebar__item--active" : ""}`}
                onClick={() => handleMenuClick(key)}
              >
                <Icon size={18} />
                <span>{label}</span>
              </button>
            ))}
          </nav>
          <button type="button" className="sd-sidebar__logout" onClick={handleLogout}>
            <LogOut size={18} />
            <span>Logout</span>
          </button>
        </aside>

        {/* ==================================================
            MOBILE SIDEBAR
        ================================================== */}
        {sidebarOpen && (
          <div className="sd-drawer-overlay" onClick={() => setSidebarOpen(false)}>
            <aside className="sd-sidebar sd-sidebar--mobile" onClick={(e) => e.stopPropagation()}>
              <div className="sd-sidebar__mobile-head">
                <span className="sd-header__brand-text">Menu</span>
                <button
                  type="button"
                  style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
                  onClick={() => setSidebarOpen(false)}
                  aria-label="Close menu"
                >
                  <X size={18} />
                </button>
              </div>

              <nav className="sd-sidebar__nav">
                {MENU_ITEMS.map(({ key, label, icon: Icon }) => (
                  <button
                    type="button"
                    key={key}
                    className={`sd-sidebar__item ${activeMenu === key ? "sd-sidebar__item--active" : ""}`}
                    onClick={() => handleMenuClick(key)}
                  >
                    <Icon size={18} />
                    <span>{label}</span>
                  </button>
                ))}
              </nav>

              <button
                type="button"
                className="sd-sidebar__logout"
                onClick={() => {
                  setSidebarOpen(false);
                  handleLogout();
                }}
              >
                <LogOut size={18} />
                <span>Logout</span>
              </button>
            </aside>
          </div>
        )}

        {/* ==================================================
            MAIN CONTENT
        ================================================== */}
        <main className="sd-main">
          {activeMenu === "profile" ? (
            <ProfileContent studentData={studentData} />
          ) : (
            <>
              <h1 className="sd-welcome">Welcome, {studentData.name} 👋</h1>
              <p className="sd-welcome-sub">
                Manage your classes, schedule and attendance from one place.
              </p>

              {/* OVERVIEW CARDS */}
              <section className="sd-overview">
                {overviewCards.map((card) => (
                  <OverviewCard key={card.key} {...card} />
                ))}
              </section>

              {/* DASHBOARD SCHEDULE  */}
              {(activeMenu === "dashboard" || activeMenu === "schedule") && (
                <section className="sd-card sd-schedule-section">
                  <div className="sd-schedule-section__head">
                    <div>
                      <h2>Classes Schedule</h2>
                      <p>Keep track of your classes routines and attendance tracking times.</p>
                      <span className="sd-today">{TODAY_LABEL}</span>
                    </div>
                  </div>

                  <div className="sd-schedule-list">
                    {schedule.map((item) => (
                      <ScheduleItem key={item.id} item={item} />
                    ))}
                  </div>
                </section>
              )}

              {/* ATTENDANCE & REPORT section visible on dashboard, history, reports  */}
              {(activeMenu === "dashboard" || activeMenu === "history" || activeMenu === "reports") && (
                <section className="sd-card sd-recent">
                  <h2>Recent Attendance / Grades</h2>
                  <RecentAttendanceTable rows={attendanceRows} />
                </section>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
