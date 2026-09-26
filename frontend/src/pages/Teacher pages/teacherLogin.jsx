import React, { useState } from "react";
import PageLogo from "../../assets/cctvLogo.png";
import {
  Presentation,
  IdCard,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import api from "../../api/axious";
import "./TeacherLogin.css";

export default function TeacherLogin({ onGoHome }) {
  const navigate = useNavigate();

  const [teacherId, setTeacherId] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!teacherId.trim() || !password.trim()) {
      setError("Please enter both Teacher ID and Password.");
      return;
    }

    try {
  setLoading(true);

  const res = await api.post("/auth/teacher/login", {
    teacherId: teacherId.trim(),
    password,
  });

  const token = res.data.data.token;
  const teacher = res.data.data.teacher;

  if (!token) {
    throw new Error("Login succeeded but no token was returned by the server.");
  }

  localStorage.setItem("token", token);

  if (teacher) {
    localStorage.setItem("teacher", JSON.stringify(teacher));
  }

  navigate("/teacher-dashboard");

} catch (err) {
  setError(getLoginErrorMessage(err));
} finally {
  setLoading(false);
  }
};

  return (
    <div className="tl-page">
      <div className="tl-bg" />

      <button type="button" className="tl-back" onClick={onGoHome}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <div className="tl-card">
        <div className="tl-card__brand">
          <div className="cas-navbar__logo">
            <img src={PageLogo} alt="CCTV Logo" className="cas-navbar__logo-img" />
          </div>
          <span className="tl-card__brand-text">CCTV ATTENDANCE SYSTEM</span>
        </div>

        <div className="tl-card__icon">
          <Presentation size={30} />
        </div>

        <h1 className="tl-card__title">Teacher Login</h1>
        <p className="tl-card__subtitle">
          Access your dashboard and manage attendance
        </p>

        {error && <div className="tl-error">{error}</div>}

        <form className="tl-form" onSubmit={handleSubmit} noValidate>
          <label className="tl-field">
            <span className="tl-field__label">Teacher ID</span>
            <div className="tl-field__control">
              <IdCard size={18} className="tl-field__icon" />
              <input
                type="text"
                placeholder="Enter your Teacher ID"
                value={teacherId}
                onChange={(e) => setTeacherId(e.target.value)}
                autoComplete="username"
              />
            </div>
          </label>

          <label className="tl-field">
            <span className="tl-field__label">Password</span>
            <div className="tl-field__control">
              <Lock size={18} className="tl-field__icon" />
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
              <button
                type="button"
                className="tl-field__toggle"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </label>

          <div className="tl-form__row">
            <label className="tl-checkbox">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />
              Remember me
            </label>
            <a href="#forgot-password" className="tl-link">
              Forgot password?
            </a>
          </div>

          <button type="submit" className="tl-submit" disabled={loading}>
            {loading ? "Signing in..." : "Login as Teacher"}
            {!loading && <ArrowRight size={16} />}
          </button>
        </form>

        <p className="tl-register-prompt">
          Don't have an account?{" "}
          <Link to="/teacher-register" className="tr-link tr-link--strong">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}

/**
 * Maps backend error responses to a user-friendly message.
 * Same "no authController.js available" caveat as TeacherRegister.jsx —
 * update the `data?.message` lookup if your controller uses a different key.
 */
function getLoginErrorMessage(err) {
  if (!err.response) {
    return "Unable to reach the server. Please check your connection and try again.";
  }

  const { status, data } = err.response;

  if (data?.errors?.length) {
    return data.errors[0].msg || data.errors[0].message || "Please check your details and try again.";
  }
  if (data?.message) {
    return data.message;
  }

  if (status === 401) return "Invalid Teacher ID or password.";
  if (status === 404) return "No account found with this Teacher ID.";
  if (status === 500) return "Server error. Please try again later.";

  return "Something went wrong. Please try again.";
}
