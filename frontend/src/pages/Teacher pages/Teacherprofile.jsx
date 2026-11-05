import React, { useEffect, useState } from "react";
import { Mail, Building2, IdCard, CheckCircle2 } from "lucide-react";
import "./TeacherProfile.css";
import api from "../../api/axious";

/* Mock data — swap for GET /api/teacher/profile later */
const INITIAL_PROFILE = {
  firstName: "",
  lastName: "",
  teacherId: "",
  email: "",
  department: "",
};

const DEPARTMENTS = [
  "Computer Science & Design",
  "Computer Science & Engineering",
  "Information Technology",
  "Electronics & Communication",
  "Mechanical Engineering",
  "Civil Engineering",
];

function ToggleSwitch({ checked, onChange, label, description }) {
  return (
    <div className="tp-toggle-row">
      <div>
        <strong>{label}</strong>
        {description && <p>{description}</p>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        className={`tp-switch ${checked ? "tp-switch--on" : ""}`}
        onClick={() => onChange(!checked)}
      >
        <span className="tp-switch__thumb" />
      </button>
    </div>
  );
}

export default function TeacherProfile({ onProfileUpdated }) {
  const [profile, setProfile] = useState(INITIAL_PROFILE);
  const [savedMessage, setSavedMessage] = useState("");

  const [loading, setLoading] = useState(true);
  const [profileError, setProfileError] = useState("");

  const [passwords, setPasswords] = useState({ current: "", next: "", confirm: "" });
  const [passwordError, setPasswordError] = useState("");
  const [passwordSuccess, setPasswordSuccess] = useState("");

  const [attendanceNotifs, setAttendanceNotifs] = useState(true);
  const [lowAttendanceAlerts, setLowAttendanceAlerts] = useState(true);

  useEffect(() => {
  const fetchProfile = async () => {
    try {
      setLoading(true);
      setProfileError("");

      const response = await api.get("/auth/teacher/profile");

      const teacher = response.data.data.teacher;

      setProfile({
        firstName: teacher.firstName || "",
        lastName: teacher.lastName || "",
        teacherId: teacher.teacherId || "",
        email: teacher.email || "",
        department: teacher.department || "",
      });
    } catch (error) {
      console.error("Failed to load teacher profile:", error);
      setProfileError(
        error.response?.data?.message || "Failed to load profile."
      );
    } finally {
      setLoading(false);
    }
  };

  fetchProfile();
}, []);
  const handleProfileChange = (field) => (e) =>
    setProfile((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSaveProfile = async (e) => {
    e.preventDefault();

    try {
      setSavedMessage("");

      const response = await api.put("/auth/teacher/profile", {
        firstName: profile.firstName,
        lastName: profile.lastName,
        email: profile.email,
        department: profile.department,
      });
      
      const updatedTeacher = response.data.data.teacher;

      setProfile({
        firstName: updatedTeacher.firstName || "",
        lastName: updatedTeacher.lastName || "",
        teacherId: updatedTeacher.teacherId || "",
        email: updatedTeacher.email || "",
        department: updatedTeacher.department || "",
      });
      if (onProfileUpdated) {
        onProfileUpdated(
          `${updatedTeacher.firstName || ""} ${
            updatedTeacher.lastName || ""
          }`.trim()
        );
      }

      // Keep localStorage teacher data updated
      localStorage.setItem("teacher", JSON.stringify(updatedTeacher));

      setSavedMessage("Profile updated successfully.");

      setTimeout(() => setSavedMessage(""), 3000);
    } catch (error) {
      console.error("Failed to update teacher profile:", error);

      setSavedMessage(
        error.response?.data?.message || "Failed to update profile."
      );
    }
  };

  const handlePasswordChange = (field) => (e) =>
    setPasswords((prev) => ({ ...prev, [field]: e.target.value }));

  const handleChangePassword = (e) => {
    e.preventDefault();
    setPasswordError("");
    setPasswordSuccess("");

    if (!passwords.current || !passwords.next || !passwords.confirm) {
      setPasswordError("Please fill in all password fields.");
      return;
    }
    if (passwords.next.length < 6) {
      setPasswordError("New password must be at least 6 characters.");
      return;
    }
    if (passwords.next !== passwords.confirm) {
      setPasswordError("New password and confirm password do not match.");
      return;
    }

    setPasswordSuccess("Password changed successfully.");
    setPasswords({ current: "", next: "", confirm: "" });
    setTimeout(() => setPasswordSuccess(""), 3000);
  };

  const initials = `${profile.firstName[0] || ""}${profile.lastName[0] || ""}`.toUpperCase();

  return (
    <div className="tp-page">
      {loading && <p className="tp-message">Loading profile...</p>}

      {profileError && (
        <div className="tp-alert tp-alert--error">
          {profileError}
        </div>
      )}
      <div className="tp-header">
        <h1>Profile &amp; Settings</h1>
        <p>Manage your teacher account information.</p>
      </div>

      <div className="tp-card tp-summary">
        <span className="tp-avatar">{initials}</span>
        <div className="tp-summary__info">
          <h2>{profile.firstName} {profile.lastName}</h2>
          <div className="tp-summary__meta">
            <span><IdCard size={14} /> Teacher ID: {profile.teacherId}</span>
            <span><Building2 size={14} /> {profile.department}</span>
            <span><Mail size={14} /> {profile.email}</span>
          </div>
        </div>
      </div>

      <div className="tp-card">
        <h3 className="tp-section-title">Personal Information</h3>
        <form className="tp-form" onSubmit={handleSaveProfile}>
          <div className="tp-form__grid">
            <label className="tp-field">
              <span>First Name</span>
              <input type="text" value={profile.firstName} onChange={handleProfileChange("firstName")} />
            </label>
            <label className="tp-field">
              <span>Last Name</span>
              <input type="text" value={profile.lastName} onChange={handleProfileChange("lastName")} />
            </label>
          </div>

          <div className="tp-form__grid">
            <label className="tp-field">
              <span>Teacher ID</span>
              <input type="text" value={profile.teacherId} disabled />
            </label>
            <label className="tp-field">
              <span>Email</span>
              <input type="email" value={profile.email} onChange={handleProfileChange("email")} />
            </label>
          </div>

          <label className="tp-field">
            <span>Department</span>
            <select value={profile.department} onChange={handleProfileChange("department")}>
              {DEPARTMENTS.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>

          <button type="submit" className="tp-btn tp-btn--green">Save Changes</button>
          {savedMessage && (
            <p className="tp-message tp-message--success"><CheckCircle2 size={15} /> {savedMessage}</p>
          )}
        </form>
      </div>

      <div className="tp-card">
        <h3 className="tp-section-title">Account Settings</h3>
        <p className="tp-section-sub">Change Password</p>

        <form className="tp-form" onSubmit={handleChangePassword}>
          {passwordError && <div className="tp-alert tp-alert--error">{passwordError}</div>}
          {passwordSuccess && (
            <div className="tp-alert tp-alert--success">
              <CheckCircle2 size={15} /> {passwordSuccess}
            </div>
          )}

          <label className="tp-field">
            <span>Current Password</span>
            <input type="password" value={passwords.current} onChange={handlePasswordChange("current")} />
          </label>

          <div className="tp-form__grid">
            <label className="tp-field">
              <span>New Password</span>
              <input type="password" value={passwords.next} onChange={handlePasswordChange("next")} />
            </label>
            <label className="tp-field">
              <span>Confirm Password</span>
              <input type="password" value={passwords.confirm} onChange={handlePasswordChange("confirm")} />
            </label>
          </div>

          <button type="submit" className="tp-btn tp-btn--green">Change Password</button>
        </form>
      </div>

      <div className="tp-card">
        <h3 className="tp-section-title">Notification Settings</h3>

        <ToggleSwitch
          checked={attendanceNotifs}
          onChange={setAttendanceNotifs}
          label="Attendance Notifications"
          description="Get notified when an attendance session starts or ends."
        />
        <ToggleSwitch
          checked={lowAttendanceAlerts}
          onChange={setLowAttendanceAlerts}
          label="Low Attendance Alerts"
          description="Get notified when a student falls below the attendance threshold."
        />
      </div>
    </div>
  );
}