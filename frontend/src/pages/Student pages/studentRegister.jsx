import { useState } from "react";
import PageLogo from "../../assets/cctvLogo.png";
import {
  GraduationCap,
  User,
  IdCard,
  Lock,
  Mail,
  Eye,
  EyeOff,
  Building2,
  BookOpen,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import api from "../../api/axious";
import "../Teacher pages/teacherRegister.css";

const DEPARTMENTS = [
  "Computer Science & Engineering",
  "Computer Science and Design",
  "Artificial Intelligence and Data Science",
  "Information Technology",
  "Electronics & Communication",
  "Electrical Engineering",
  "Mechanical Engineering",
  "Civil Engineering",
  "Management Studies",
];

export default function StudentRegister() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    studentId: "",
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    department: "",
    classDivision: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleChange = (field) => (event) => {
    setForm((previous) => ({ ...previous, [field]: event.target.value }));
  };

  const validate = () => {
    if (Object.values(form).some((value) => !value.trim())) {
      return "Please fill in all fields.";
    }
    if (!/^\S+@\S+\.\S+$/.test(form.email.trim())) {
      return "Please enter a valid email address.";
    }
    if (form.password.length < 8 || !/[A-Z]/.test(form.password) ||
        !/[a-z]/.test(form.password) || !/[0-9]/.test(form.password)) {
      return "Password must be 8+ characters and include uppercase, lowercase, and a number.";
    }
    if (form.password !== form.confirmPassword) {
      return "Passwords do not match.";
    }
    return "";
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess(false);
    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    try {
      setLoading(true);
      const payload = { ...form };
      delete payload.confirmPassword;
      await api.post("/auth/student/register", payload);
      setSuccess(true);
      setForm({
        studentId: "",
        firstName: "",
        lastName: "",
        email: "",
        password: "",
        confirmPassword: "",
        department: "",
        classDivision: "",
      });
      setTimeout(() => navigate("/student-login"), 1200);
    } catch (err) {
      setError(getStudentRegisterErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tr-page">
      <div className="tr-bg" />
      <button type="button" className="tr-back" onClick={() => navigate("/")}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <div className="tr-card">
        <div className="tr-card__brand">
          <div className="cas-navbar__logo">
            <img src={PageLogo} alt="CCTV Logo" className="cas-navbar__logo-img" />
          </div>
          <span className="tr-card__brand-text">CCTV ATTENDANCE SYSTEM</span>
        </div>
        <div className="tr-card__icon"><GraduationCap size={30} /></div>
        <h1 className="tr-card__title">Student Registration</h1>
        <p className="tr-card__subtitle">Create an account linked to your existing face record</p>

        {error && <div className="tr-alert tr-alert--error">{error}</div>}
        {success && (
          <div className="tr-alert tr-alert--success">
            Account created successfully! Redirecting to login...
          </div>
        )}

        <form className="tr-form" onSubmit={handleSubmit} noValidate>
          <label className="tr-field">
            <span className="tr-field__label">Student ID</span>
            <div className="tr-field__control">
              <IdCard size={18} className="tr-field__icon" />
              <input type="text" placeholder="e.g. STU001" value={form.studentId}
                onChange={handleChange("studentId")} autoComplete="username" />
            </div>
          </label>

          <div className="tr-form__grid">
            <StudentTextField label="First Name" icon={User} placeholder="e.g. Anjali"
              value={form.firstName} onChange={handleChange("firstName")} autoComplete="given-name" />
            <StudentTextField label="Last Name" icon={User} placeholder="e.g. Sharma"
              value={form.lastName} onChange={handleChange("lastName")} autoComplete="family-name" />
          </div>

          <StudentTextField label="Email" icon={Mail} placeholder="you@college.edu"
            type="email" value={form.email} onChange={handleChange("email")} autoComplete="email" />

          <div className="tr-form__grid">
            <label className="tr-field">
              <span className="tr-field__label">Department</span>
              <div className="tr-field__control">
                <Building2 size={18} className="tr-field__icon" />
                <select value={form.department} onChange={handleChange("department")}>
                  <option value="" disabled>Select department</option>
                  {DEPARTMENTS.map((department) => <option key={department} value={department}>{department}</option>)}
                </select>
              </div>
            </label>
            <StudentTextField label="Class / Division" icon={BookOpen} placeholder="e.g. TY-A"
              value={form.classDivision} onChange={handleChange("classDivision")} />
          </div>

          <div className="tr-form__grid">
            <PasswordField label="Create Password" value={form.password} show={showPassword}
              onChange={handleChange("password")} onToggle={() => setShowPassword((value) => !value)} />
            <PasswordField label="Confirm Password" value={form.confirmPassword} show={showConfirm}
              onChange={handleChange("confirmPassword")} onToggle={() => setShowConfirm((value) => !value)} />
          </div>

          <button type="submit" className="tr-submit" disabled={loading}>
            {loading ? "Creating account..." : "Create Account"}
            {!loading && <ArrowRight size={16} />}
          </button>
        </form>

        <p className="tr-login-prompt">
          Already have an account?{" "}
          <Link to="/student-login" className="tr-link tr-link--strong">Log in</Link>
        </p>
      </div>
    </div>
  );
}

function StudentTextField({ label, icon: Icon, type = "text", ...props }) {
  return (
    <label className="tr-field">
      <span className="tr-field__label">{label}</span>
      <div className="tr-field__control">
        <Icon size={18} className="tr-field__icon" />
        <input type={type} {...props} />
      </div>
    </label>
  );
}

function PasswordField({ label, value, show, onChange, onToggle }) {
  return (
    <label className="tr-field">
      <span className="tr-field__label">{label}</span>
      <div className="tr-field__control">
        <Lock size={18} className="tr-field__icon" />
        <input type={show ? "text" : "password"} placeholder="8+ chars, Aa1"
          value={value} onChange={onChange} autoComplete="new-password" />
        <button type="button" className="tr-field__toggle" onClick={onToggle}
          aria-label={show ? "Hide password" : "Show password"}>
          {show ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>
    </label>
  );
}

function getStudentRegisterErrorMessage(error) {
  if (!error.response) {
    return "Unable to reach the server. Please check your connection and try again.";
  }
  if (error.response.data?.errors?.length) {
    return error.response.data.errors[0].message || "Please check the form and try again.";
  }
  return error.response.data?.message || "Unable to register. Please try again.";
}
