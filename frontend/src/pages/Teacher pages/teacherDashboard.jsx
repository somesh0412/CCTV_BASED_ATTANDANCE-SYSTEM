import React, { useState } from "react";
import {
  ShieldCheck,
  Menu,
  X,
  Bell,
  ChevronDown,
  LayoutDashboard,
  CalendarClock,
  Video,
  History,
  Users,
  FileBarChart,
  Settings,
  LogOut,
  Plus,
  Clock,
  MapPin,
  PlayCircle,
  Eye,
  GraduationCap,
  UserCheck,
  UserX,
  TrendingUp,
} from "lucide-react";

import "./TeacherDashboard.css";
import LiveAttendance from "./LiveAttendance";


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
    label: "Today's Schedule",
    icon: CalendarClock,
  },
  {
    key: "live",
    label: "Live Attendance",
    icon: Video,
  },
  {
    key: "history",
    label: "Attendance History",
    icon: History,
  },
  {
    key: "students",
    label: "Students",
    icon: Users,
  },
  {
    key: "reports",
    label: "Reports",
    icon: FileBarChart,
  },
  {
    key: "profile",
    label: "Profile / Settings",
    icon: Settings,
  },
];


const OVERVIEW_CARDS = [
  {
    key: "total",
    label: "Total Students",
    value: 42,
    icon: GraduationCap,
    tone: "blue",
  },
  {
    key: "present",
    label: "Present Today",
    value: 38,
    icon: UserCheck,
    tone: "green",
  },
  {
    key: "absent",
    label: "Absent Today",
    value: 4,
    icon: UserX,
    tone: "red",
  },
  {
    key: "rate",
    label: "Attendance Rate",
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


const RECENT_ATTENDANCE = [
  {
    id: "r1",
    subject: "Machine Learning",
    classDivision: "TY CSD-D",
    date: "15 Aug",
    present: 38,
    absent: 4,
    status: "Completed",
  },
  {
    id: "r2",
    subject: "DBMS",
    classDivision: "TY CSD-D",
    date: "14 Aug",
    present: 40,
    absent: 2,
    status: "Completed",
  },
  {
    id: "r3",
    subject: "Computer Networks",
    classDivision: "TY CSD-D",
    date: "13 Aug",
    present: 36,
    absent: 6,
    status: "Completed",
  },
];


const SUBJECT_OPTIONS = [
  "Machine Learning",
  "Database Management System",
  "Computer Networks",
  "Operating Systems",
  "Software Engineering",
];


const CLASS_OPTIONS = [
  "TY CSD - A",
  "TY CSD - B",
  "TY CSD - C",
  "TY CSD - D",
];


const TODAY_LABEL = new Date().toLocaleDateString(
  "en-US",
  {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  }
);


/* ----------------------------------------------------------
   Overview Card
---------------------------------------------------------- */

function OverviewCard({
  icon: Icon,
  label,
  value,
  tone,
}) {
  return (
    <div className="td-overview-card">

      <span
        className={`td-overview-card__icon td-overview-card__icon--${tone}`}
      >
        <Icon size={20} />
      </span>

      <div>

        <p className="td-overview-card__label">
          {label}
        </p>

        <strong className="td-overview-card__value">
          {value}
        </strong>

      </div>

    </div>
  );
}


/* ----------------------------------------------------------
   Schedule Item
---------------------------------------------------------- */

function ScheduleItem({
  item,
  onStartAttendance,
  sessionStarted,
}) {

  const isLive = item.status === "live";

  return (
    <div
      className={`td-schedule-item ${
        isLive
          ? "td-schedule-item--live"
          : ""
      }`}
    >

      <div className="td-schedule-item__time">

        <Clock size={16} />

        <span>
          {item.startTime} – {item.endTime}
        </span>

      </div>


      <div className="td-schedule-item__info">

        <h4>
          {item.subject}
        </h4>

        <p>

          {item.classDivision}

          <span className="td-dot">
            &bull;
          </span>

          <MapPin size={13} />

          {item.room}

        </p>

      </div>


      <div className="td-schedule-item__status">

        <span
          className={`td-badge ${
            isLive
              ? "td-badge--live"
              : "td-badge--upcoming"
          }`}
        >
          {isLive
            ? "Live Now"
            : "Upcoming"}
        </span>

      </div>


      <div className="td-schedule-item__action">

        {isLive ? (

          <button
            type="button"
            className="td-btn td-btn--green"
            onClick={() =>
              onStartAttendance(item.id)
            }
          >

            <PlayCircle size={16} />

            {sessionStarted
              ? "Session Started"
              : "Start Attendance"}

          </button>

        ) : (

          <button
            type="button"
            className="td-btn td-btn--outline"
          >

            <Eye size={16} />

            View Details

          </button>

        )}

      </div>

    </div>
  );
}


/* ----------------------------------------------------------
   Add Class Modal
---------------------------------------------------------- */

function AddClassModal({
  open,
  onClose,
  onAdd,
}) {

  const [form, setForm] = useState({
    subject: "",
    classDivision: "",
    startTime: "",
    endTime: "",
    room: "",
  });


  if (!open) {
    return null;
  }


  const handleChange =
    (field) =>
    (e) => {

      setForm((prev) => ({
        ...prev,
        [field]: e.target.value,
      }));

    };


  const handleSubmit = (e) => {

    e.preventDefault();

    if (
      !form.subject ||
      !form.classDivision ||
      !form.startTime ||
      !form.endTime ||
      !form.room
    ) {
      return;
    }

    onAdd(form);

    setForm({
      subject: "",
      classDivision: "",
      startTime: "",
      endTime: "",
      room: "",
    });
  };


  const formatTime = (value) => {

    if (!value) {
      return "";
    }

    const [h, m] =
      value.split(":");

    const hour =
      ((+h + 11) % 12) + 1;

    const ampm =
      +h >= 12
        ? "PM"
        : "AM";

    return `${hour}:${m} ${ampm}`;
  };


  return (
    <div
      className="td-modal-overlay"
      onClick={onClose}
    >

      <div
        className="td-modal"
        onClick={(e) =>
          e.stopPropagation()
        }
      >

        <div className="td-modal__header">

          <h3>
            Add Class
          </h3>

          <button
            type="button"
            className="td-modal__close"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={18} />
          </button>

        </div>


        <form
          className="td-modal__form"
          onSubmit={handleSubmit}
        >

          <label className="td-field">

            <span>
              Subject
            </span>

            <select
              value={form.subject}
              onChange={handleChange("subject")}
            >

              <option
                value=""
                disabled
              >
                Select Subject
              </option>

              {SUBJECT_OPTIONS.map(
                (subject) => (
                  <option
                    key={subject}
                    value={subject}
                  >
                    {subject}
                  </option>
                )
              )}

            </select>

          </label>


          <label className="td-field">

            <span>
              Class / Division
            </span>

            <select
              value={form.classDivision}
              onChange={handleChange(
                "classDivision"
              )}
            >

              <option
                value=""
                disabled
              >
                Select Class
              </option>

              {CLASS_OPTIONS.map(
                (className) => (
                  <option
                    key={className}
                    value={className}
                  >
                    {className}
                  </option>
                )
              )}

            </select>

          </label>


          <div className="td-modal__row">

            <label className="td-field">

              <span>
                Start Time
              </span>

              <input
                type="time"
                value={form.startTime}
                onChange={handleChange(
                  "startTime"
                )}
              />

            </label>


            <label className="td-field">

              <span>
                End Time
              </span>

              <input
                type="time"
                value={form.endTime}
                onChange={handleChange(
                  "endTime"
                )}
              />

            </label>

          </div>


          <label className="td-field">

            <span>
              Classroom
            </span>

            <input
              type="text"
              placeholder="e.g. Room 302"
              value={form.room}
              onChange={handleChange(
                "room"
              )}
            />

          </label>


          <div className="td-modal__actions">

            <button
              type="button"
              className="td-btn td-btn--outline"
              onClick={onClose}
            >
              Cancel
            </button>


            <button
              type="submit"
              className="td-btn td-btn--green"
            >
              Add Class
            </button>

          </div>


          {form.startTime &&
            form.endTime && (

              <p className="td-modal__preview">

                Preview:{" "}

                {formatTime(
                  form.startTime
                )}

                {" – "}

                {formatTime(
                  form.endTime
                )}

              </p>

            )}

        </form>

      </div>

    </div>
  );
}


/* ----------------------------------------------------------
   Recent Attendance
---------------------------------------------------------- */

function RecentAttendanceTable({
  rows,
}) {

  return (
    <>

      <table className="td-table">

        <thead>

          <tr>

            <th>
              Subject
            </th>

            <th>
              Class
            </th>

            <th>
              Date
            </th>

            <th>
              Present
            </th>

            <th>
              Absent
            </th>

            <th>
              Status
            </th>

          </tr>

        </thead>


        <tbody>

          {rows.map(
            (row) => (

              <tr key={row.id}>

                <td>
                  {row.subject}
                </td>

                <td>
                  {row.classDivision}
                </td>

                <td>
                  {row.date}
                </td>

                <td className="td-table__present">
                  {row.present}
                </td>

                <td className="td-table__absent">
                  {row.absent}
                </td>

                <td>

                  <span className="td-badge td-badge--completed">
                    {row.status}
                  </span>

                </td>

              </tr>

            )
          )}

        </tbody>

      </table>


      <div className="td-table-cards">

        {rows.map(
          (row) => (

            <div
              className="td-table-card"
              key={row.id}
            >

              <div className="td-table-card__head">

                <strong>
                  {row.subject}
                </strong>

                <span className="td-badge td-badge--completed">
                  {row.status}
                </span>

              </div>


              <p className="td-table-card__meta">

                {row.classDivision}

                &bull;

                {row.date}

              </p>


              <div className="td-table-card__stats">

                <span className="td-table__present">
                  Present: {row.present}
                </span>

                <span className="td-table__absent">
                  Absent: {row.absent}
                </span>

              </div>

            </div>

          )
        )}

      </div>

    </>
  );
}


/* ----------------------------------------------------------
   MAIN COMPONENT
---------------------------------------------------------- */

export default function TeacherDashboard({
  teacherName = "Prof. Sharma",
  onLogout,
  onNavigate,
}) {

  const [
    sidebarOpen,
    setSidebarOpen,
  ] = useState(false);


  const [
    activeMenu,
    setActiveMenu,
  ] = useState("dashboard");


  const [
    profileOpen,
    setProfileOpen,
  ] = useState(false);


  const [
    schedule,
    setSchedule,
  ] = useState(
    INITIAL_SCHEDULE
  );


  const [
    modalOpen,
    setModalOpen,
  ] = useState(false);


  const [
    startedSessions,
    setStartedSessions,
  ] = useState({});


  /* ========================================================
     MENU NAVIGATION
  ======================================================== */

  const handleMenuClick = (
    key
  ) => {

    setActiveMenu(key);

    setSidebarOpen(false);

    if (onNavigate) {
      onNavigate(key);
    }
  };


  /* ========================================================
     ADD CLASS
  ======================================================== */

  const handleAddClass = (
    form
  ) => {

    const formatTime = (
      value
    ) => {

      const [h, m] =
        value.split(":");

      const hour =
        ((+h + 11) % 12) + 1;

      const ampm =
        +h >= 12
          ? "PM"
          : "AM";

      return `${hour}:${m} ${ampm}`;
    };


    const newItem = {

      id: `s${Date.now()}`,

      startTime:
        formatTime(
          form.startTime
        ),

      endTime:
        formatTime(
          form.endTime
        ),

      subject:
        form.subject,

      classDivision:
        form.classDivision,

      room:
        form.room,

      status:
        "upcoming",
    };


    setSchedule(
      (prev) => [
        ...prev,
        newItem,
      ]
    );


    setModalOpen(false);
  };


  /* ========================================================
     START ATTENDANCE FROM SCHEDULE
  ======================================================== */

  const handleStartAttendance = (
    id
  ) => {

    setStartedSessions(
      (prev) => ({
        ...prev,
        [id]: true,
      })
    );

    /*
      Also open Live Attendance page.
    */

    handleMenuClick("live");
  };


  return (

    <div className="td-page">


      {/* ====================================================
          HEADER
      ==================================================== */}

      <header className="td-header">

        <div className="td-header__left">

          <button
            type="button"
            className="td-hamburger"
            onClick={() =>
              setSidebarOpen(true)
            }
            aria-label="Open menu"
          >
            <Menu size={22} />
          </button>


          <div className="td-header__brand">

            <span className="td-header__brand-icon">
              <ShieldCheck size={20} />
            </span>

            <span className="td-header__brand-text">
              CCTV ATTENDANCE SYSTEM
            </span>

          </div>

        </div>


        <div className="td-header__right">

          <button
            type="button"
            className="td-icon-btn"
            aria-label="Notifications"
          >

            <Bell size={19} />

            <span className="td-icon-btn__dot" />

          </button>


          <div className="td-profile">

            <button
              type="button"
              className="td-profile__trigger"
              onClick={() =>
                setProfileOpen(
                  (v) => !v
                )
              }
            >

              <span className="td-profile__avatar">

                {teacherName
                  .split(" ")
                  .map(
                    (w) => w[0]
                  )
                  .slice(0, 2)
                  .join("")}

              </span>


              <span className="td-profile__name">
                {teacherName}
              </span>


              <ChevronDown size={16} />

            </button>


            {profileOpen && (

              <div className="td-profile__dropdown">

                <button
                  type="button"
                  onClick={() =>
                    handleMenuClick(
                      "profile"
                    )
                  }
                >

                  <Settings size={15} />

                  Profile / Settings

                </button>


                <button
                  type="button"
                  onClick={onLogout}
                  className="td-profile__logout"
                >

                  <LogOut size={15} />

                  Logout

                </button>

              </div>

            )}

          </div>

        </div>

      </header>


      <div className="td-body">


        {/* ==================================================
            DESKTOP SIDEBAR
        ================================================== */}

        <aside className="td-sidebar td-sidebar--desktop">

          <nav className="td-sidebar__nav">

            {MENU_ITEMS.map(
              ({
                key,
                label,
                icon: Icon,
              }) => (

                <button
                  type="button"
                  key={key}
                  className={`td-sidebar__item ${
                    activeMenu === key
                      ? "td-sidebar__item--active"
                      : ""
                  }`}
                  onClick={() =>
                    handleMenuClick(
                      key
                    )
                  }
                >

                  <Icon size={18} />

                  <span>
                    {label}
                  </span>

                </button>

              )
            )}

          </nav>


          <button
            type="button"
            className="td-sidebar__logout"
            onClick={onLogout}
          >

            <LogOut size={18} />

            <span>
              Logout
            </span>

          </button>

        </aside>


        {/* ==================================================
            MOBILE SIDEBAR
        ================================================== */}

        {sidebarOpen && (

          <div
            className="td-drawer-overlay"
            onClick={() =>
              setSidebarOpen(false)
            }
          >

            <aside
              className="td-sidebar td-sidebar--mobile"
              onClick={(e) =>
                e.stopPropagation()
              }
            >

              <div className="td-sidebar__mobile-head">

                <span className="td-header__brand-text">
                  Menu
                </span>


                <button
                  type="button"
                  className="td-modal__close"
                  onClick={() =>
                    setSidebarOpen(false)
                  }
                  aria-label="Close menu"
                >
                  <X size={18} />
                </button>

              </div>


              <nav className="td-sidebar__nav">

                {MENU_ITEMS.map(
                  ({
                    key,
                    label,
                    icon: Icon,
                  }) => (

                    <button
                      type="button"
                      key={key}
                      className={`td-sidebar__item ${
                        activeMenu === key
                          ? "td-sidebar__item--active"
                          : ""
                      }`}
                      onClick={() =>
                        handleMenuClick(
                          key
                        )
                      }
                    >

                      <Icon size={18} />

                      <span>
                        {label}
                      </span>

                    </button>

                  )
                )}

              </nav>


              <button
                type="button"
                className="td-sidebar__logout"
                onClick={() => {

                  setSidebarOpen(false);

                  if (onLogout) {
                    onLogout();
                  }

                }}
              >

                <LogOut size={18} />

                <span>
                  Logout
                </span>

              </button>

            </aside>

          </div>

        )}


        {/* ==================================================
            MAIN CONTENT
        ================================================== */}

        <main className="td-main">


          {activeMenu === "live" ? (

            /*
              ================================================
              LIVE ATTENDANCE PAGE
              ================================================
            */

            <LiveAttendance />

          ) : (

            /*
              ================================================
              NORMAL DASHBOARD PAGE
              ================================================
            */

            <>

              <h1 className="td-welcome">

                Welcome, {teacherName} 👋

              </h1>


              <p className="td-welcome-sub">

                Manage your classes,
                schedule and attendance
                from one place.

              </p>


              {/* ==================================================
                  OVERVIEW CARDS
              ================================================== */}

              <section className="td-overview">

                {OVERVIEW_CARDS.map(
                  (card) => (

                    <OverviewCard
                      key={card.key}
                      {...card}
                    />

                  )
                )}

              </section>


              {/* ==================================================
                  SCHEDULE
              ================================================== */}

              <section className="td-card td-schedule-section">

                <div className="td-schedule-section__head">

                  <div>

                    <h2>
                      Schedule Your Day
                    </h2>

                    <p>
                      Plan your classes
                      and manage attendance
                      sessions.
                    </p>

                    <span className="td-today">
                      {TODAY_LABEL}
                    </span>

                  </div>


                  <button
                    type="button"
                    className="td-btn td-btn--green"
                    onClick={() =>
                      setModalOpen(true)
                    }
                  >

                    <Plus size={16} />

                    Add Class

                  </button>

                </div>


                <div className="td-schedule-list">

                  {schedule.map(
                    (item) => (

                      <ScheduleItem
                        key={item.id}
                        item={item}
                        onStartAttendance={
                          handleStartAttendance
                        }
                        sessionStarted={
                          !!startedSessions[
                            item.id
                          ]
                        }
                      />

                    )
                  )}

                </div>

              </section>


              {/* ==================================================
                  QUICK ACTIONS
              ================================================== */}

              <section className="td-card td-quick-actions">

                <h2>
                  Quick Actions
                </h2>


                <div className="td-quick-actions__grid">


                  <button
                    type="button"
                    className="td-btn td-btn--outline"
                    onClick={() =>
                      handleMenuClick(
                        "live"
                      )
                    }
                  >

                    <Video size={16} />

                    Start Live Attendance

                  </button>


                  <button
                    type="button"
                    className="td-btn td-btn--outline"
                    onClick={() =>
                      handleMenuClick(
                        "history"
                      )
                    }
                  >

                    <History size={16} />

                    View Attendance History

                  </button>


                  <button
                    type="button"
                    className="td-btn td-btn--outline"
                    onClick={() =>
                      handleMenuClick(
                        "students"
                      )
                    }
                  >

                    <Users size={16} />

                    View Students

                  </button>


                  <button
                    type="button"
                    className="td-btn td-btn--outline"
                    onClick={() =>
                      handleMenuClick(
                        "reports"
                      )
                    }
                  >

                    <FileBarChart size={16} />

                    Generate Report

                  </button>

                </div>

              </section>


              {/* ==================================================
                  RECENT ATTENDANCE
              ================================================== */}

              <section className="td-card td-recent">

                <h2>
                  Recent Attendance
                </h2>


                <RecentAttendanceTable
                  rows={
                    RECENT_ATTENDANCE
                  }
                />

              </section>

            </>

          )}

        </main>

      </div>


      {/* ====================================================
          ADD CLASS MODAL
      ==================================================== */}

      <AddClassModal
        open={modalOpen}
        onClose={() =>
          setModalOpen(false)
        }
        onAdd={
          handleAddClass
        }
      />

    </div>
  );
}