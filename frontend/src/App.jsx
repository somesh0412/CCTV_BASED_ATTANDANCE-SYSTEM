import { useState } from 'react'
import { Routes, Route } from 'react-router-dom';
import './App.css'
import LandingPage from './pages/landingPage';
import TeacherLogin from './pages/Teacher pages/teacherLogin';
import TeacherRegister from './pages/Teacher pages/teacherRegister';
import TeacherDashboard from './pages/Teacher pages/teacherDashboard';
import StudentDashboard from './pages/Student pages/StudentDashboard';

function App() {
  

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/teacher-register" element={<TeacherRegister />} />
      <Route path="/teacher-login" element={<TeacherLogin />} />
      <Route path="/teacher-dashboard" element={<TeacherDashboard />} />
      <Route path="/student-dashboard" element={<StudentDashboard />} />
    </Routes>
  )
}

export default App
