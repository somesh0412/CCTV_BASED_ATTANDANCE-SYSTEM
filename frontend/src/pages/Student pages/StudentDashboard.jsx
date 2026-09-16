import React, { useState } from 'react';
import {
  Home,
  Calendar,
  BarChart3,
  LineChart,
  User,
  LogOut,
  Menu,
  X,
  Bell,
  ChevronDown,
  BookOpen,
  Users,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';
import './StudentDashboard.css';
import StudentSidebar from './StudentSidebar';

// Sidebar Component
const Sidebar = ({ activeMenu, setActiveMenu, sidebarOpen, setSidebarOpen }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'schedule', label: 'My Schedule', icon: Calendar },
    { id: 'attendance', label: 'My Attendance', icon: BarChart3 },
    { id: 'report', label: 'Attendance Report', icon: LineChart },
    { id: 'profile', label: 'Profile', icon: User }
  ];

  return (
    <>
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <AlertCircle size={24} />
            <span>CCTV ATTENDANCE SYSTEM</span>
          </div>
          <button className="close-btn" onClick={() => setSidebarOpen(false)}>
            <X size={24} />
          </button>
        </div>

        <nav className="sidebar-nav">
          {menuItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-item ${activeMenu === item.id ? 'active' : ''}`}
                onClick={() => {
                  setActiveMenu(item.id);
                  setSidebarOpen(false);
                }}
              >
                <IconComponent size={20} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <button className="nav-item logout-btn">
          <LogOut size={20} />
          <span>Logout</span>
        </button>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </>
  );
};

// Header Component
const Header = ({ sidebarOpen, setSidebarOpen, showProfileMenu, setShowProfileMenu }) => {
  return (
    <header className="header">
      <div className="header-left">
        <button
          className="hamburger-btn"
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          <Menu size={24} />
        </button>
        <div className="welcome-text">
          <h1>Welcome, Somesh 👋</h1>
          <p>Here's your attendance overview.</p>
        </div>
      </div>

      <div className="header-right">
        <button className="notification-btn">
          <Bell size={20} />
        </button>

        <div className="profile-menu-container">
          <button
            className="profile-btn"
            onClick={() => setShowProfileMenu(!showProfileMenu)}
          >
            <div className="profile-avatar">S</div>
            <div className="profile-info">
              <span className="profile-name">Somesh Raut</span>
            </div>
            <ChevronDown size={18} className={showProfileMenu ? 'rotate' : ''} />
          </button>

          {showProfileMenu && (
            <div className="profile-dropdown">
              <a href="#" className="dropdown-item">
                <User size={18} />
                View Profile
              </a>
              <a href="#" className="dropdown-item logout">
                <LogOut size={18} />
                Logout
              </a>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

// Summary Card Component
const SummaryCard = ({ icon: Icon, title, value, isPercentage = false }) => {
  return (
    <div className="summary-card">
      <div className="card-header">
        <h3>{title}</h3>
        <Icon size={24} className="card-icon" />
      </div>
      <div className="card-value">
        {value}
        {isPercentage && '%'}
      </div>
    </div>
  );
};

// Schedule Card Component
const ScheduleCard = ({ time, subject, className, room, status }) => {
  const statusColor = status === 'Present' ? 'present' : 'upcoming';
  
  return (
    <div className={`schedule-card status-${statusColor}`}>
      <div className="schedule-time">{time}</div>
      <div className="schedule-details">
        <h4>{subject}</h4>
        <p className="schedule-class">{className}</p>
        <p className="schedule-room">Room {room}</p>
      </div>
      <div className={`schedule-status ${statusColor}`}>
        {status === 'Present' ? (
          <>
            <CheckCircle size={18} />
            {status}
          </>
        ) : (
          <>
            <Clock size={18} />
            {status}
          </>
        )}
      </div>
    </div>
  );
};

// Attendance Table Component
const AttendanceTable = ({ attendanceData }) => {
  return (
    <div className="attendance-section">
      <h2>Recent Attendance</h2>
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
            {attendanceData.map((record, index) => (
              <tr key={index}>
                <td>{record.date}</td>
                <td>{record.subject}</td>
                <td>{record.class}</td>
                <td>
                  <span className={`status-badge ${record.status.toLowerCase()}`}>
                    <span className="status-dot"></span>
                    {record.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Subject Attendance Component
const SubjectAttendance = ({ subjects }) => {
  return (
    <div className="subject-attendance-section">
      <h2>Subject-wise Attendance</h2>
      <div className="subjects-grid">
        {subjects.map((subject, index) => (
          <div key={index} className="subject-card">
            <div className="subject-header">
              <h4>{subject.name}</h4>
              <span className="subject-percentage">{subject.percentage}%</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${subject.percentage}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Face Recognition Status Component
const FaceRecognitionStatus = () => {
  return (
    <div className="face-recognition-card">
      <div className="face-card-header">
        <h3>Face Recognition Profile</h3>
      </div>
      <div className="face-card-content">
        <div className="status-item">
          <span className="status-label">Status:</span>
          <div className="status-value">
            <CheckCircle size={18} className="success-icon" />
            <span>Registered</span>
          </div>
        </div>
        <div className="status-item">
          <span className="status-label">Last Updated:</span>
          <span className="status-value-text">10 August 2026</span>
        </div>
      </div>
      <button className="view-profile-btn">View Profile</button>
    </div>
  );
};

// Profile Page Component
const ProfilePage = () => {
  const profileData = {
    name: 'Somesh Raut',
    studentId: 'CSD-D-28',
    department: 'Computer Science and Design',
    class: 'TY CSD - D',
    email: 'student@college.edu'
  };

  return (
    <div className="profile-page">
      <div className="profile-header">
        <div className="profile-avatar-large">S</div>
        <h1>{profileData.name}</h1>
      </div>

      <div className="profile-info-grid">
        <div className="profile-info-item">
          <label>Name</label>
          <p>{profileData.name}</p>
        </div>
        <div className="profile-info-item">
          <label>Student ID</label>
          <p>{profileData.studentId}</p>
        </div>
        <div className="profile-info-item">
          <label>Department</label>
          <p>{profileData.department}</p>
        </div>
        <div className="profile-info-item">
          <label>Class / Division</label>
          <p>{profileData.class}</p>
        </div>
        <div className="profile-info-item">
          <label>Email</label>
          <p>{profileData.email}</p>
        </div>
      </div>
    </div>
  );
};

// Main Dashboard Component
const DashboardContent = ({ activeMenu }) => {
  // Mock Data
  const attendanceData = [
    { date: '16 Aug', subject: 'Machine Learning', class: 'TY CSD-D', status: 'Present' },
    { date: '15 Aug', subject: 'DBMS', class: 'TY CSD-D', status: 'Present' },
    { date: '14 Aug', subject: 'Computer Networks', class: 'TY CSD-D', status: 'Absent' },
    { date: '13 Aug', subject: 'Data Science', class: 'TY CSD-D', status: 'Present' }
  ];

  const scheduleData = [
    { time: '10:15 AM – 11:15 AM', subject: 'Machine Learning', className: 'TY CSD - D', room: '201', status: 'Present' },
    { time: '12:00 PM – 1:00 PM', subject: 'Database Management System', className: 'TY CSD - D', room: '304', status: 'Upcoming' },
    { time: '2:00 PM – 3:00 PM', subject: 'Computer Networks', className: 'TY CSD - D', room: '205', status: 'Upcoming' }
  ];

  const subjectData = [
    { name: 'Machine Learning', percentage: 90 },
    { name: 'Database Management System', percentage: 80 },
    { name: 'Computer Networks', percentage: 100 },
    { name: 'Data Science', percentage: 90 }
  ];

  const today = new Date().toLocaleDateString('en-US', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  });

  if (activeMenu === 'profile') {
    return <ProfilePage />;
  }

  if (activeMenu === 'schedule') {
    return (
      <div className="content">
        <h2>My Schedule</h2>
        <p className="section-date">{today}</p>
        <div className="schedule-cards">
          {scheduleData.map((schedule, index) => (
            <ScheduleCard key={index} {...schedule} />
          ))}
        </div>
      </div>
    );
  }

  if (activeMenu === 'attendance') {
    return (
      <div className="content">
        <h2>My Attendance</h2>
        <div className="attendance-overview">
          <SummaryCard icon={BookOpen} title="Total Classes" value="42" />
          <SummaryCard icon={CheckCircle} title="Classes Attended" value="38" />
          <SummaryCard icon={AlertCircle} title="Classes Missed" value="4" />
          <SummaryCard icon={BarChart3} title="Overall Attendance" value="90.5" isPercentage={true} />
        </div>
        <SubjectAttendance subjects={subjectData} />
      </div>
    );
  }

  if (activeMenu === 'report') {
    return (
      <div className="content">
        <h2>Attendance Report</h2>
        <AttendanceTable attendanceData={attendanceData} />
      </div>
    );
  }

  // Default Dashboard
  return (
    <div className="content">
      <div className="attendance-summary">
        <SummaryCard icon={BookOpen} title="Total Classes" value="42" />
        <SummaryCard icon={CheckCircle} title="Classes Attended" value="38" />
        <SummaryCard icon={AlertCircle} title="Classes Missed" value="4" />
        <SummaryCard icon={BarChart3} title="Overall Attendance" value="90.5" isPercentage={true} />
      </div>

      <section className="today-schedule-section">
        <div className="section-header">
          <h2>Today's Schedule</h2>
          <p className="section-date">{today}</p>
        </div>
        <div className="schedule-cards">
          {scheduleData.map((schedule, index) => (
            <ScheduleCard key={index} {...schedule} />
          ))}
        </div>
      </section>

      <div className="dashboard-grid">
        <SubjectAttendance subjects={subjectData} />
        <FaceRecognitionStatus />
      </div>

      <AttendanceTable attendanceData={attendanceData} />
    </div>
  );
};

// Main App Component
export default function StudentDashboard() {
  const [activeMenu, setActiveMenu] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);

  return (
    <div className="student-dashboard">
      <Sidebar
        activeMenu={activeMenu}
        setActiveMenu={setActiveMenu}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />

      <div className="main-container">
        <Header
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          showProfileMenu={showProfileMenu}
          setShowProfileMenu={setShowProfileMenu}
        />

        <main className="main-content">
          <DashboardContent activeMenu={activeMenu} />
        </main>
      </div>
    </div>
  );
}
