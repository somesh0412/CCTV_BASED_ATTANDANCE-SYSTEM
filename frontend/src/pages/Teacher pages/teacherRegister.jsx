import React, { useState } from "react";
import PageLogo from "../../assets/cctvLogo.png";
import {
  Presentation,
  User,
  IdCard,
  Lock,
  Mail,
  Eye,
  EyeOff,
  Building2,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import api from "../../api/axious";
import "./TeacherRegister.css";

const DEPARTMENTS = [
  "Computer Science & Engineering",
  "Information Technology",
  "Electronics & Communication",
  "Electrical Engineering",
  "Mechanical Engineering",
  "Civil Engineering",
  "Mathematics",
  "Physics",
  "Chemistry",
  "Management Studies",
];

export default function TeacherRegister({ onGoHome }) {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    teacherId: "",
    email: "",
    password: "",
    confirmPassword: "",
    department: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleChange = (field) => (e) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const validate = () => {
    if (
      !form.firstName.trim() ||
      !form.lastName.trim() ||
      !form.teacherId.trim() ||
      !form.email.trim() ||
      !form.password ||
      !form.confirmPassword ||
      !form.department
    ) {
      return "Please fill in all fields.";
    }
    // Basic email shape check — backend (isEmail()) does the real validation.
    if (!/^\S+@\S+\.\S+$/.test(form.email.trim())) {
      return "Please enter a valid email address.";
    }
    if (form.password.length < 6) {
      return "Password must be at least 6 characters long.";
    }
    if (form.password !== form.confirmPassword) {
      return "Passwords do not match.";
    }
    return "";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess(false);

    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    try {
      setLoading(true);

      // confirmPassword is only needed for the frontend check above —
      // don't send it to the backend.
      const { confirmPassword, ...payload } = form;

      await api.post("/auth/teacher/register", payload);

      setSuccess(true);
      setForm({
        firstName: "",
        lastName: "",
        teacherId: "",
        email: "",
        password: "",
        confirmPassword: "",
        department: "",
      });

      // Give the success message a moment to be visible, then send
      // the teacher to login (per the required Register -> Login flow).
      setTimeout(() => navigate("/teacher-login"), 1200);
    } catch (err) {
      setError(getRegisterErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tr-page">
      <div className="tr-bg" />

      <button type="button" className="tr-back" onClick={onGoHome}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <div className="tr-card">
        <div className="tr-card__brand">
          <div className="cas-navbar__logo">
            <img src={PageLogo} alt="CCTV Logo" className="cas-navbar__logo-img" />
          </div>
          <span className="tr-card__brand-text">CCTV ATTENDANCE SYSTEM</span>
        </div>

        <div className="tr-card__icon">
          <Presentation size={30} />
        </div>

        <h1 className="tr-card__title">Teacher Registration</h1>
        <p className="tr-card__subtitle">
          Create an account to access your dashboard
        </p>

        {error && <div className="tr-alert tr-alert--error">{error}</div>}
        {success && (
          <div className="tr-alert tr-alert--success">
            Account created successfully! Redirecting to login...
          </div>
        )}

        <form className="tr-form" onSubmit={handleSubmit} noValidate>
          <div className="tr-form__grid">
            <label className="tr-field">
              <span className="tr-field__label">First Name</span>
              <div className="tr-field__control">
                <User size={18} className="tr-field__icon" />
                <input
                  type="text"
                  placeholder="e.g. Anjali"
                  value={form.firstName}
                  onChange={handleChange("firstName")}
                  autoComplete="given-name"
                />
              </div>
            </label>

            <label className="tr-field">
              <span className="tr-field__label">Last Name</span>
              <div className="tr-field__control">
                <User size={18} className="tr-field__icon" />
                <input
                  type="text"
                  placeholder="e.g. Sharma"
                  value={form.lastName}
                  onChange={handleChange("lastName")}
                  autoComplete="family-name"
                />
              </div>
            </label>
          </div>

          <label className="tr-field">
            <span className="tr-field__label">Teacher ID</span>
            <div className="tr-field__control">
              <IdCard size={18} className="tr-field__icon" />
              <input
                type="text"
                placeholder="Enter your Teacher ID"
                value={form.teacherId}
                onChange={handleChange("teacherId")}
                autoComplete="username"
              />
            </div>
          </label>

          {/*
            Added: backend registerValidation requires "email" but the
            original form didn't collect it — registration would always
            fail without this field. Styled identically to the other
            tr-field inputs so the design is unchanged.
          */}
          <label className="tr-field">
            <span className="tr-field__label">Email</span>
            <div className="tr-field__control">
              <Mail size={18} className="tr-field__icon" />
              <input
                type="email"
                placeholder="you@college.edu"
                value={form.email}
                onChange={handleChange("email")}
                autoComplete="email"
              />
            </div>
          </label>

          <label className="tr-field">
            <span className="tr-field__label">Department</span>
            <div className="tr-field__control">
              <Building2 size={18} className="tr-field__icon" />
              <select
                value={form.department}
                onChange={handleChange("department")}
              >
                <option value="" disabled>
                  Select your department
                </option>
                {DEPARTMENTS.map((dept) => (
                  <option key={dept} value={dept}>
                    {dept}
                  </option>
                ))}
              </select>
            </div>
          </label>

          <div className="tr-form__grid">
            <label className="tr-field">
              <span className="tr-field__label">Create Password</span>
              <div className="tr-field__control">
                <Lock size={18} className="tr-field__icon" />
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Min. 6 characters"
                  value={form.password}
                  onChange={handleChange("password")}
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  className="tr-field__toggle"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </label>

            <label className="tr-field">
              <span className="tr-field__label">Confirm Password</span>
              <div className="tr-field__control">
                <Lock size={18} className="tr-field__icon" />
                <input
                  type={showConfirm ? "text" : "password"}
                  placeholder="Re-enter password"
                  value={form.confirmPassword}
                  onChange={handleChange("confirmPassword")}
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  className="tr-field__toggle"
                  onClick={() => setShowConfirm((v) => !v)}
                  aria-label={showConfirm ? "Hide password" : "Show password"}
                >
                  {showConfirm ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </label>
          </div>

          <button type="submit" className="tr-submit" disabled={loading}>
            {loading ? "Creating account..." : "Create Account"}
            {!loading && <ArrowRight size={16} />}
          </button>
        </form>

        <p className="tr-login-prompt">
          Already have an account?{" "}
          <Link to="/teacher-login" className="tr-link tr-link--strong">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
}

/**
 * Maps backend error responses to a user-friendly message.
 *
 * ASSUMPTION FLAGGED FOR REVIEW (no authController.js was available):
 * - express-validator (via validateMiddleware) commonly returns
 *   `{ errors: [{ msg: "..." }, ...] }` on 400 — handled below.
 * - A custom controller error is assumed to return `{ message: "..." }`.
 * If your authController.js uses a different shape (e.g. `{ error: "..." }`
 * or `{ success: false, msg: "..." }`), update the two lookups below —
 * everything else in this file stays the same.
 */
function getRegisterErrorMessage(err) {
  if (!err.response) {
    return "Unable to reach the server. Please check your connection and try again.";
  }

  const { status, data } = err.response;

  if (data?.errors?.length) {
    return data.errors[0].msg || data.errors[0].message || "Please check the form and try again.";
  }
  if (data?.message) {
    return data.message;
  }

  if (status === 409) return "A teacher with this Teacher ID or email already exists.";
  if (status === 400) return "Please check the form and try again.";
  if (status === 500) return "Server error. Please try again later.";

  return "Something went wrong. Please try again.";
}