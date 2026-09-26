import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
  Home,
  Calendar,
  Camera,
  ClipboardCheck,
  BarChart3,
  User,
  LogOut,
  Menu,
  X,
} from 'lucide-react';
import './StudentSidebar.css';

// Configuration
const STUDENT_LOGIN_ROUTE = '/student-login'; // Change this if your login route is different

const StudentSidebar = ({ activeItem = 'dashboard', onLogout = null }) => {
  const navigate = useNavigate();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [showLogoutModal, setShowLogoutModal] = useState(false);

  // Menu items configuration
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home, route: '/student-dashboard' },
    { id: 'schedule', label: 'My Schedule', icon: Calendar, route: '/student-schedule' },
    { id: 'face-registration', label: 'Face Registration', icon: Camera, route: '/student-face-registration' },
    { id: 'attendance', label: 'My Attendance', icon: ClipboardCheck, route: '/student-attendance' },
    { id: 'attendance-report', label: 'Attendance Report', icon: BarChart3, route: '/student-attendance-report' },
    { id: 'profile', label: 'Profile', icon: User, route: '/student-profile' },
  ];

  // Handle logout confirmation
  const handleLogoutConfirm = () => {
    // Clear authentication token
    localStorage.removeItem('authToken');
    sessionStorage.removeItem('authToken');

    // Call custom logout handler if provided
    if (onLogout) {
      onLogout();
    }

    // Close modals and menu
    setShowLogoutModal(false);
    setIsMobileMenuOpen(false);

    // Navigate to login page
    navigate(STUDENT_LOGIN_ROUTE);
  };

  // Handle navigation on mobile
  const handleNavClick = () => {
    setIsMobileMenuOpen(false);
  };

  // Handle logout button click
  const handleLogoutClick = () => {
    setShowLogoutModal(true);
  };

  return (
    <>
      {/* Mobile Header */}
      <div className="mobile-header">
        <button
          className="hamburger-btn"
          onClick={() => setIsMobileMenuOpen(true)}
          aria-label="Open navigation menu"
        >
          <Menu size={24} />
        </button>
        <h1 className="mobile-header-title">Student Dashboard</h1>
      </div>

      {/* Sidebar */}
      <aside className={`student-sidebar ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
        {/* Close button (mobile only) */}
        <button
          className="sidebar-close-btn"
          onClick={() => setIsMobileMenuOpen(false)}
          aria-label="Close navigation menu"
        >
          <X size={24} />
        </button>

        {/* Logo Section */}
        <div className="sidebar-logo">
          <div className="logo-circle">◉</div>
          <div className="logo-text">
            <div>CCTV ATTENDANCE</div>
            <div>SYSTEM</div>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="sidebar-nav">
          {menuItems.map((item) => {
            const IconComponent = item.icon;
            const isActive = activeItem === item.id;

            return (
              <NavLink
                key={item.id}
                to={item.route}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={handleNavClick}
                aria-current={isActive ? 'page' : undefined}
              >
                <IconComponent size={20} className="nav-icon" />
                <span className="nav-label">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Logout Button */}
        <div className="sidebar-footer">
          <button
            className="logout-btn"
            onClick={handleLogoutClick}
            aria-label="Logout from student portal"
          >
            <LogOut size={20} className="nav-icon" />
            <span className="nav-label">Logout</span>
          </button>
        </div>
      </aside>

      {/* Mobile Overlay */}
      {isMobileMenuOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setIsMobileMenuOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Logout Confirmation Modal */}
      {showLogoutModal && (
        <>
          <div
            className="modal-overlay"
            onClick={() => setShowLogoutModal(false)}
            aria-hidden="true"
          />
          <div className="logout-modal" role="dialog" aria-labelledby="logout-modal-title">
            <h2 id="logout-modal-title" className="modal-title">
              Confirm Logout
            </h2>
            <p className="modal-message">
              Are you sure you want to logout?
            </p>
            <div className="modal-buttons">
              <button
                className="modal-btn cancel-btn"
                onClick={() => setShowLogoutModal(false)}
              >
                Cancel
              </button>
              <button
                className="modal-btn logout-confirm-btn"
                onClick={handleLogoutConfirm}
              >
                Logout
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default StudentSidebar;
