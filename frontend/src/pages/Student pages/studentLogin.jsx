import { useState } from "react";
import PageLogo from "../../assets/cctvLogo.png";
import {
  GraduationCap,
  IdCard,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import api from "../../api/axious";
import "../Teacher pages/teacherLogin.css";

export default function StudentLogin() {
  const navigate = useNavigate();
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    if (!identifier.trim() || !password) {
      setError("Please enter your Student ID/email and password.");
      return;
    }

    try {
      setLoading(true);
      const response = await api.post("/auth/student/login", {
        identifier: identifier.trim(),
        password,
      });
      const { token, student } = response.data.data;

      if (!token || !student) {
        throw new Error("Login succeeded but student authentication data was not returned.");
      }

      localStorage.setItem("token", token);
      localStorage.setItem("student", JSON.stringify(student));
      localStorage.removeItem("teacher");
      navigate(student.faceRegistered ? "/student-dashboard" : "/student-face-registration");
    } catch (err) {
      setError(getStudentLoginErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tl-page">
      <div className="tl-bg" />
      <button type="button" className="tl-back" onClick={() => navigate("/")}>
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
          <GraduationCap size={30} />
        </div>
        <h1 className="tl-card__title">Student Login</h1>
        <p className="tl-card__subtitle">Access your attendance and student profile</p>

        {error && <div className="tl-error">{error}</div>}

        <form className="tl-form" onSubmit={handleSubmit} noValidate>
          <label className="tl-field">
            <span className="tl-field__label">Student ID or Email</span>
            <div className="tl-field__control">
              <IdCard size={18} className="tl-field__icon" />
              <input
                type="text"
                placeholder="Enter your Student ID or email"
                value={identifier}
                onChange={(event) => setIdentifier(event.target.value)}
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
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
              />
              <button
                type="button"
                className="tl-field__toggle"
                onClick={() => setShowPassword((value) => !value)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </label>

          <button type="submit" className="tl-submit" disabled={loading}>
            {loading ? "Signing in..." : "Login as Student"}
            {!loading && <ArrowRight size={16} />}
          </button>
        </form>

        <p className="tl-register-prompt">
          Do not have an account?{" "}
          <Link to="/student-register" className="tr-link tr-link--strong">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}

function getStudentLoginErrorMessage(error) {
  if (!error.response) {
    return "Unable to reach the server. Please check your connection and try again.";
  }
  if (error.response.data?.errors?.length) {
    return error.response.data.errors[0].message || "Please check your login details.";
  }
  if (error.response.status === 404) {
    return "Student account not found. Please register first.";
  }
  return error.response.data?.message || "Unable to log in. Please try again.";
}
