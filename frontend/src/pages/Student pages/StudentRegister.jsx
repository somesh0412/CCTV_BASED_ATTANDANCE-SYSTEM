import { useState, useEffect, useRef } from "react";
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
    ArrowRight,
    ArrowLeft,
    Camera,
    CheckCircle2,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import api from "../../api/axious";
import "./StudentRegister.css";

const FACE_ENGINE_URL = import.meta.env.VITE_FACE_ENGINE_URL || "http://localhost:8000";

const DEPARTMENTS = [
    "Computer Science & Engineering",
    "Information Technology",
    "Computer Science and Design",
    "Electronics & Communication",
    "Mechanical Engineering",
];

const INSTRUCTIONS = [
    { text: "Please look straight into the camera.", time: 3000 },
    { text: "Great! Now please smile slightly.", time: 3000 },
    { text: "Turn your face slowly to the left.", time: 3000 },
    { text: "Now turn your face slowly to the right.", time: 3000 },
    { text: "Look slightly upwards.", time: 3000 },
    { text: "Capturing clear images. Hang tight...", time: 2000 },
];

export default function StudentRegister() {
    const navigate = useNavigate();

    const [form, setForm] = useState({
        firstName: "",
        lastName: "",
        studentId: "",
        email: "",
        password: "",
        confirmPassword: "",
        department: "",
    });

    const [showPassword, setShowPassword] = useState(false);
    const [showConfirm, setShowConfirm] = useState(false);
    const [error, setError] = useState("");

    // Registration flow state
    const [step, setStep] = useState("form"); // "form" | "camera" | "success"
    const [cameraInstructionIndex, setCameraInstructionIndex] = useState(0);
    const [instructionCompleted, setInstructionCompleted] = useState(false);

    // WebRTC stream states
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const capturedFramesRef = useRef([]);
    const [stream, setStream] = useState(null);

    const handleChange = (field) => (e) => {
        setForm((prev) => ({ ...prev, [field]: e.target.value }));
    };

    const validate = () => {
        if (
            !form.firstName.trim() ||
            !form.lastName.trim() ||
            !form.studentId.trim() ||
            !form.email.trim() ||
            !form.password ||
            !form.confirmPassword ||
            !form.department
        ) {
            return "Please fill in all fields.";
        }
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

    const proceedToCamera = (e) => {
        e.preventDefault();
        setError("");

        const validationError = validate();
        if (validationError) {
            setError(validationError);
            return;
        }

        setStep("camera");
    };

    useEffect(() => {
        let mediaStream = null;

        const startCamera = async () => {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
                setStream(mediaStream);
                if (videoRef.current) {
                    videoRef.current.srcObject = mediaStream;
                }
            } catch (err) {
                console.error("Camera access denied or unavailable", err);
                setError("Unable to access camera. Please check permissions.");
            }
        };

        if (step === "camera") {
            startCamera();
        }

        return () => {
            if (mediaStream) {
                mediaStream.getTracks().forEach((track) => track.stop());
            }
        };
    }, [step]);

    const captureFrame = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        if (capturedFramesRef.current.length < 6) {
            capturedFramesRef.current.push(canvas.toDataURL("image/jpeg", 0.85));
        }
    };

    useEffect(() => {
        if (step === "camera" && stream && cameraInstructionIndex < INSTRUCTIONS.length) {
            const timer = setTimeout(() => {
                captureFrame();
                if (cameraInstructionIndex === INSTRUCTIONS.length - 1) {
                    setInstructionCompleted(true);
                } else {
                    setCameraInstructionIndex((prev) => prev + 1);
                }
            }, INSTRUCTIONS[cameraInstructionIndex].time);
            return () => clearTimeout(timer);
        }
    }, [step, cameraInstructionIndex, stream]);

    const finishRegistration = async () => {
        setError("");

        try {
            captureFrame();
            const frames = capturedFramesRef.current;
            if (frames.length === 0) {
                throw new Error("No camera frame was captured. Please try again.");
            }

            const faceEmbeddings = [];
            for (const image of frames) {
                const response = await fetch(`${FACE_ENGINE_URL}/register/embedding`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image }),
                });
                const result = await response.json();
                if (!response.ok) continue;
                faceEmbeddings.push(result.embedding);
            }

            if (faceEmbeddings.length === 0) {
                throw new Error("No face was detected. Please retake the registration photos.");
            }

            const payload = { ...form };
            delete payload.confirmPassword;
            payload.faceEmbeddings = faceEmbeddings;
            await api.post("/auth/student/register", payload);

            setStep("success");

            setTimeout(() => navigate("/student-login"), 1500);
        } catch (err) {
            setError(err.response?.data?.message || err.message || "Registration failed. Please try again.");
            setStep("form");
        }
    };

    return (
        <div className="sr-page">
            <div className="sr-bg" />

            <button type="button" className="sr-back" onClick={() => navigate("/")}>
                <ArrowLeft size={16} /> Back to Home
            </button>

            <div className="sr-card">
                <div className="sr-card__brand">
                    <div className="cas-navbar__logo">
                        <img src={PageLogo} alt="CCTV Logo" className="cas-navbar__logo-img" />
                    </div>
                    <span className="sr-card__brand-text">CCTV ATTENDANCE SYSTEM</span>
                </div>

                {step === "form" && (
                    <>
                        <div className="sr-card__icon">
                            <GraduationCap size={30} />
                        </div>

                        <h1 className="sr-card__title">Student Registration</h1>
                        <p className="sr-card__subtitle">
                            Create an account & register your facial details
                        </p>

                        {error && <div className="sr-alert sr-alert--error">{error}</div>}

                        <form className="sr-form" onSubmit={proceedToCamera} noValidate>
                            <div className="sr-form__grid">
                                <label className="sr-field">
                                    <span className="sr-field__label">First Name</span>
                                    <div className="sr-field__control">
                                        <User size={18} className="sr-field__icon" />
                                        <input
                                            type="text"
                                            placeholder="e.g. John"
                                            value={form.firstName}
                                            onChange={handleChange("firstName")}
                                        />
                                    </div>
                                </label>

                                <label className="sr-field">
                                    <span className="sr-field__label">Last Name</span>
                                    <div className="sr-field__control">
                                        <User size={18} className="sr-field__icon" />
                                        <input
                                            type="text"
                                            placeholder="e.g. Doe"
                                            value={form.lastName}
                                            onChange={handleChange("lastName")}
                                        />
                                    </div>
                                </label>
                            </div>

                            <label className="sr-field">
                                <span className="sr-field__label">Student ID / Roll No</span>
                                <div className="sr-field__control">
                                    <IdCard size={18} className="sr-field__icon" />
                                    <input
                                        type="text"
                                        placeholder="e.g. CSD-D-28"
                                        value={form.studentId}
                                        onChange={handleChange("studentId")}
                                    />
                                </div>
                            </label>

                            <label className="sr-field">
                                <span className="sr-field__label">Email</span>
                                <div className="sr-field__control">
                                    <Mail size={18} className="sr-field__icon" />
                                    <input
                                        type="email"
                                        placeholder="student@college.edu"
                                        value={form.email}
                                        onChange={handleChange("email")}
                                    />
                                </div>
                            </label>

                            <label className="sr-field">
                                <span className="sr-field__label">Department / Branch</span>
                                <div className="sr-field__control">
                                    <Building2 size={18} className="sr-field__icon" />
                                    <select
                                        value={form.department}
                                        onChange={handleChange("department")}
                                    >
                                        <option value="" disabled>Select your department</option>
                                        {DEPARTMENTS.map((dept) => (
                                            <option key={dept} value={dept}>{dept}</option>
                                        ))}
                                    </select>
                                </div>
                            </label>

                            <div className="sr-form__grid">
                                <label className="sr-field">
                                    <span className="sr-field__label">Create Password</span>
                                    <div className="sr-field__control">
                                        <Lock size={18} className="sr-field__icon" />
                                        <input
                                            type={showPassword ? "text" : "password"}
                                            placeholder="Min. 6 characters"
                                            value={form.password}
                                            onChange={handleChange("password")}
                                        />
                                        <button
                                            type="button"
                                            className="sr-field__toggle"
                                            onClick={() => setShowPassword((v) => !v)}
                                        >
                                            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                                        </button>
                                    </div>
                                </label>

                                <label className="sr-field">
                                    <span className="sr-field__label">Confirm</span>
                                    <div className="sr-field__control">
                                        <Lock size={18} className="sr-field__icon" />
                                        <input
                                            type={showConfirm ? "text" : "password"}
                                            placeholder="Re-enter password"
                                            value={form.confirmPassword}
                                            onChange={handleChange("confirmPassword")}
                                        />
                                        <button
                                            type="button"
                                            className="sr-field__toggle"
                                            onClick={() => setShowConfirm((v) => !v)}
                                        >
                                            {showConfirm ? <EyeOff size={18} /> : <Eye size={18} />}
                                        </button>
                                    </div>
                                </label>
                            </div>

                            <button type="submit" className="sr-submit sr-submit--next">
                                Proceed to Camera <ArrowRight size={16} />
                            </button>
                        </form>

                        <p className="sr-login-prompt">
                            Already have an account?{" "}
                            <Link to="/student-login" className="sr-link sr-link--strong">
                                Log in
                            </Link>
                        </p>
                    </>
                )}

                {step === "camera" && (
                    <div className="sr-camera-view">
                        <h1 className="sr-card__title">Capture Face Details</h1>
                        <p className="sr-card__subtitle">Follow instructions to complete registration</p>

                        <div className="sr-webcam-container">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="sr-webcam-video"
                            />
                            <canvas ref={canvasRef} hidden />
                            {!stream && (
                                <div className="sr-webcam-placeholder">
                                    <Camera size={48} className="sr-webcam-placeholder__icon" />
                                    <p>Accessing camera...</p>
                                </div>
                            )}
                            {stream && !instructionCompleted && (
                                <div className="sr-webcam-mock__scanning-line"></div>
                            )}
                        </div>

                        <div className="sr-instruction-box">
                            <p className="sr-instruction-text">
                                {INSTRUCTIONS[cameraInstructionIndex].text}
                            </p>
                            {!instructionCompleted && stream && (
                                <div className="sr-instruction-progress">
                                    <div
                                        className="sr-instruction-progress-bar"
                                        style={{
                                            animationDuration: `${INSTRUCTIONS[cameraInstructionIndex].time}ms`,
                                            animationName: "loading-bar"
                                        }}
                                    ></div>
                                </div>
                            )}
                        </div>

                        <button
                            type="button"
                            className="sr-submit"
                            disabled={!instructionCompleted}
                            onClick={finishRegistration}
                        >
                            {instructionCompleted ? "Complete Registration" : "Capturing Data..."}
                            {instructionCompleted && <CheckCircle2 size={16} />}
                        </button>
                    </div>
                )}

                {step === "success" && (
                    <div className="sr-success-view">
                        <div className="sr-card__icon sr-card__icon--success">
                            <CheckCircle2 size={48} />
                        </div>
                        <h1 className="sr-card__title">Registration Successful!</h1>
                        <p className="sr-card__subtitle">Your profile & face details are saved securely.</p>
                        <p className="sr-login-prompt">Redirecting to login...</p>
                    </div>
                )}
            </div>
        </div>
    );
}
