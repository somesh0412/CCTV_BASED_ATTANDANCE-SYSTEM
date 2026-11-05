import { Navigate, Outlet } from "react-router-dom";

export default function ProtectedStudentRoute() {
  const token = localStorage.getItem("token");
  const student = getStoredStudent();

  if (!token || student?.role !== "student") {
    return <Navigate to="/student-login" replace />;
  }

  return <Outlet />;
}

function getStoredStudent() {
  try {
    return JSON.parse(localStorage.getItem("student") || "null");
  } catch {
    return null;
  }
}
