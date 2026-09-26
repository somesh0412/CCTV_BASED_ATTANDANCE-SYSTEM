import React from "react";
import PageLogo from "../assets/cctvLogo.png";
import { useNavigate } from "react-router-dom";
import {
  ShieldCheck,
  Moon,
  Sparkles,
  Target,
  Clock,
  GraduationCap,
  Presentation,
  ArrowRight,
  FileText,
  Star,
  CheckCircle2,
  Users,
  ShieldCheck as ShieldIcon,
  Database,
  Home,
  UserRound,
  BarChart2,
  Layers,
  Settings,
  Leaf,
} from "lucide-react";
import "./landing.css";
import { Link } from "react-router-dom";

export default function LandingPage() {
  const navigate = useNavigate();
  return (
    <div className="cas-page">
      {/* ===== NAVBAR ===== */}
      <header className="cas-navbar">
        <div className="cas-navbar__brand">
          <div className="cas-navbar__logo">
            <img src={PageLogo} alt="CCTV Logo" className="cas-navbar__logo-img" />
          </div>
          <div>
            <h1 className="cas-navbar__title">
              CCTV <span>ATTENDANCE</span> SYSTEM
            </h1>
            <p className="cas-navbar__subtitle">Smart &bull; Secure &bull; Automated</p>
          </div>
        </div>

        <nav className="cas-navbar__links">
          <a href="#home" className="active">Home</a>
          <a href="#about">About</a>
          <a href="#features">Features</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#contact">Contact</a>
        </nav>

        <button className="cas-navbar__toggle" aria-label="Toggle dark mode">
          <Moon size={18} />
        </button>
      </header>

      {/* ===== HERO ===== */}
      <section className="cas-hero" id="home">
        <div className="cas-hero__bg" />

        <div className="cas-hero__left">
          <span className="cas-badge">
            <Sparkles size={14} />
            AI POWERED <b>ATTENDANCE</b>
          </span>

          <h2 className="cas-hero__title">
            Smart Attendance
            <br />
            with <span>CCTV &amp; AI</span>
          </h2>

          <p className="cas-hero__desc">
            Our system uses advanced AI and facial recognition to
            automatically mark attendance, ensuring accuracy, security, and
            efficiency.
          </p>

          <div className="cas-hero__stats">
            <div className="cas-stat">
              <span className="cas-stat__icon"><Target size={20} /></span>
              <div>
                <strong>99%+</strong>
                <p>Accuracy</p>
              </div>
            </div>
            <div className="cas-stat">
              <span className="cas-stat__icon"><ShieldIcon size={20} /></span>
              <div>
                <strong>Secure</strong>
                <p>Data Protection</p>
              </div>
            </div>
            <div className="cas-stat">
              <span className="cas-stat__icon"><Clock size={20} /></span>
              <div>
                <strong>Real-time</strong>
                <p>Attendance</p>
              </div>
            </div>
          </div>
        </div>

        <div className="cas-hero__right">
          <div className="cas-login-card">
            <div className="cas-login-card__header">
              <h3>Choose your login to continue</h3>
              <ShieldCheck size={16} className="cas-login-card__header-icon" />
            </div>

            <div className="cas-login-card__options">
              <div className="cas-login-option cas-login-option--student">
                <span className="cas-login-option__icon cas-login-option__icon--blue">
                  <GraduationCap size={28} />
                </span>
                <h4>Student Login</h4>
                <p>Access your attendance and profile</p>
                <button className="cas-btn cas-btn--blue" onClick={() => navigate('/student-login')}>
                  Login as Student <ArrowRight size={16} />
                </button>
              </div>

              <div className="cas-login-option cas-login-option--teacher">
                <span className="cas-login-option__icon cas-login-option__icon--green">
                  <Presentation size={28} />
                </span>
                <h4>Teacher Login</h4>
                <p>Access dashboard and manage attendance</p>
                <button className="cas-btn cas-btn--green" onClick={() => navigate('/teacher-login')}>
                  Login as Teacher <ArrowRight size={16} />
                </button>
              </div>
            </div>

            <div className="cas-login-card__footer">
              <span><ShieldIcon size={14} /> Secure login</span>
              <span>&bull;</span>
              <span>Protected data</span>
              <span>&bull;</span>
              <span>Trusted by institutions</span>
            </div>
          </div>
        </div>
      </section>

      {/* ===== ABOUT + FEATURES ===== */}
      <section className="cas-about" id="about">
        <div className="cas-about__left">
          <span className="cas-eyebrow">
            <FileText size={14} /> OVERVIEW
          </span>
          <h2 className="cas-section-title">
            About <span>CCTV</span> Attendance System
          </h2>
          <p className="cas-about__desc">
            The CCTV Attendance System is an AI-powered solution that uses
            facial recognition technology to automate the attendance process
            in classrooms and institutions. It eliminates manual work,
            prevents proxy attendance, and provides real-time reports to
            students and faculty.
          </p>

          <div className="cas-about__grid">
            <div className="cas-mini-card">
              <span className="cas-mini-card__icon cas-mini-card__icon--blue">
                <Users size={20} />
              </span>
              <strong>Automated</strong>
              <p>No manual entry</p>
            </div>
            <div className="cas-mini-card">
              <span className="cas-mini-card__icon cas-mini-card__icon--green">
                <CheckCircle2 size={20} />
              </span>
              <strong>Accurate</strong>
              <p>AI face recognition</p>
            </div>
            <div className="cas-mini-card">
              <span className="cas-mini-card__icon cas-mini-card__icon--orange">
                <Clock size={20} />
              </span>
              <strong>Real-time</strong>
              <p>Instant updates</p>
            </div>
            <div className="cas-mini-card">
              <span className="cas-mini-card__icon cas-mini-card__icon--purple">
                <Database size={20} />
              </span>
              <strong>Secure</strong>
              <p>Encrypted data</p>
            </div>
          </div>
        </div>

        <div className="cas-about__right" id="features">
          <span className="cas-eyebrow">
            <Star size={14} /> KEY FEATURES
          </span>

          <ul className="cas-feature-list">
            <li>
              <CheckCircle2 size={20} className="cas-feature-list__check" />
              <div>
                <strong>Real-time Face Recognition</strong>
                <p>Detects and recognizes faces in real-time using AI.</p>
              </div>
            </li>
            <li>
              <CheckCircle2 size={20} className="cas-feature-list__check" />
              <div>
                <strong>Automatic Attendance</strong>
                <p>Attendance is marked automatically without manual intervention.</p>
              </div>
            </li>
            <li>
              <CheckCircle2 size={20} className="cas-feature-list__check" />
              <div>
                <strong>Live Monitoring</strong>
                <p>Teachers can monitor attendance live from their dashboard.</p>
              </div>
            </li>
            <li>
              <CheckCircle2 size={20} className="cas-feature-list__check" />
              <div>
                <strong>Reports &amp; Analytics</strong>
                <p>Generate detailed attendance reports with insights and analytics.</p>
              </div>
            </li>
          </ul>

          <div className="cas-dashboard-mock">
            <div className="cas-dashboard-mock__sidebar">
              <Home size={16} />
              <UserRound size={16} />
              <BarChart2 size={16} />
              <Layers size={16} />
              <Settings size={16} />
            </div>
            <div className="cas-dashboard-mock__body">
              <div className="cas-dashboard-mock__topbar">
                <span>Dashboard</span>
              </div>
              <div className="cas-dashboard-mock__cards">
                <div>
                  <p>Total Students</p>
                  <strong>320</strong>
                </div>
                <div>
                  <p>Present Today</p>
                  <strong className="green">278</strong>
                </div>
                <div>
                  <p>Attendance %</p>
                  <strong className="blue">86.9%</strong>
                </div>
              </div>
              <div className="cas-dashboard-mock__chart">
                <div className="cas-dashboard-mock__chart-head">
                  <span>Attendance Overview</span>
                  <span className="cas-dashboard-mock__pill">This Week</span>
                </div>
                <div className="cas-dashboard-mock__chart-body">
                  <svg viewBox="0 0 160 60" className="cas-dashboard-mock__line">
                    <polyline
                      points="0,40 25,30 50,45 75,20 100,32 125,15 150,25"
                      fill="none"
                      stroke="var(--cas-blue)"
                      strokeWidth="2.5"
                    />
                  </svg>
                  <div className="cas-dashboard-mock__bars">
                    <span style={{ height: "40%" }} />
                    <span style={{ height: "70%" }} />
                    <span style={{ height: "55%" }} />
                    <span style={{ height: "85%" }} />
                  </div>
                  <div className="cas-dashboard-mock__donut" />
                </div>
              </div>
            </div>
            <Leaf className="cas-dashboard-mock__plant" size={40} />
          </div>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="cas-footer">
        <p>
          <ShieldCheck size={16} />
          Empowering institutions with smart technology for a better tomorrow.
        </p>
        <p className="cas-footer__copy">
          &copy; {new Date().getFullYear()} CCTV Attendance System. All rights reserved.
        </p>
      </footer>
    </div>
  );
}