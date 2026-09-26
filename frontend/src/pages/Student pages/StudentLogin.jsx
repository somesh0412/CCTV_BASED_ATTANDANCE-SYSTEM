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
import "./StudentLogin.css";

export default function StudentLogin() {
    const navigate = useNavigate();

    const [studentId, setStudentId] = useState("");
    const [password, setPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    const [remember, setRemember] = useState(false);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        if (!studentId.trim() || !password.trim()) {
            setError("Please enter both Student ID and Password.");
            return;
        }

        try {
            setLoading(true);
            const response = await api.post("/auth/student/login", {
                studentId,
                password,
            });
            const { token, student } = response.data.data;
            localStorage.setItem("token", token);
            localStorage.setItem("student", JSON.stringify(student));
            navigate("/student-dashboard");
        } catch (err) {
            setError(getLoginErrorMessage(err));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="sl-page">
            <div className="sl-bg" />

            <button type="button" className="sl-back" onClick={() => navigate("/")}>
                <ArrowLeft size={16} /> Back to Home
            </button>

            <div className="sl-card">
                <div className="sl-card__brand">
                    <div className="cas-navbar__logo">
                        <img src={PageLogo} alt="CCTV Logo" className="cas-navbar__logo-img" />
                    </div>
                    <span className="sl-card__brand-text">CCTV ATTENDANCE SYSTEM</span>
                </div>

                <div className="sl-card__icon">
                    <GraduationCap size={30} />
                </div>

                <h1 className="sl-card__title">Student Login</h1>
                <p className="sl-card__subtitle">
                    Access your attendance dashboard and profile
                </p>

                {error && <div className="sl-error">{error}</div>}

                <form className="sl-form" onSubmit={handleSubmit} noValidate>
                    <label className="sl-field">
                        <span className="sl-field__label">Student ID</span>
                        <div className="sl-field__control">
                            <IdCard size={18} className="sl-field__icon" />
                            <input
                                type="text"
                                placeholder="Enter your Student ID"
                                value={studentId}
                                onChange={(e) => setStudentId(e.target.value)}
                                autoComplete="username"
                            />
                        </div>
                    </label>

                    <label className="sl-field">
                        <span className="sl-field__label">Password</span>
                        <div className="sl-field__control">
                            <Lock size={18} className="sl-field__icon" />
                            <input
                                type={showPassword ? "text" : "password"}
                                placeholder="Enter your password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                autoComplete="current-password"
                            />
                            <button
                                type="button"
                                className="sl-field__toggle"
                                onClick={() => setShowPassword((v) => !v)}
                                aria-label={showPassword ? "Hide password" : "Show password"}
                            >
                                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                            </button>
                        </div>
                    </label>

                    <div className="sl-form__row">
                        <label className="sl-checkbox">
                            <input
                                type="checkbox"
                                checked={remember}
                                onChange={(e) => setRemember(e.target.checked)}
                            />
                            Remember me
                        </label>
                        <a href="#forgot-password" className="sl-link">
                            Forgot password?
                        </a>
                    </div>

                    <button type="submit" className="sl-submit" disabled={loading}>
                        {loading ? "Signing in..." : "Login as Student"}
                        {!loading && <ArrowRight size={16} />}
                    </button>
                </form>

                <p className="sl-register-prompt">
                    Don't have an account?{" "}
                    <Link to="/student-register" className="sr-link sr-link--strong">
                        Register
                    </Link>
                </p>
            </div>
        </div>
    );
}

function getLoginErrorMessage(err) {
    if (!err.response) {
        return "Unable to connect to server. Please try again.";
    }

    const { status, data } = err.response;

    if (data?.errors?.length) {
        return data.errors[0].msg || data.errors[0].message || "Please check your details and try again.";
    }
    if (data?.message) {
        return data.message;
    }

    if (status === 401) return "Invalid Student ID or password.";
    if (status === 404) return "No account found with this Student ID.";
    if (status === 500) return "Server error. Please try again later.";

    return "Login failed. Please check your credentials.";
}
