const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");

const studentSchema = new mongoose.Schema(
  {
    firstName: { type: String, required: true, trim: true, minlength: 2, maxlength: 50 },
    lastName: { type: String, required: true, trim: true, minlength: 1, maxlength: 50 },
    name: { type: String, required: true, trim: true },
    studentId: { type: String, required: true, unique: true, trim: true, uppercase: true },
    department: { type: String, required: true, trim: true },
    email: { type: String, required: true, unique: true, trim: true, lowercase: true },
    password: { type: String, required: true, minlength: 6, select: false },
    faceRegistered: { type: Boolean, default: false },
    faceEmbeddings: { type: [[Number]], default: [] },
    role: { type: String, enum: ["student"], default: "student" },
  },
  { timestamps: { createdAt: true, updatedAt: false } }
);

studentSchema.pre("save", async function () {
  if (!this.isModified("password")) return;
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
});

studentSchema.methods.matchPassword = function matchPassword(enteredPassword) {
  return bcrypt.compare(enteredPassword, this.password);
};

studentSchema.set("toJSON", {
  transform: (_doc, ret) => {
    delete ret.password;
    delete ret.__v;
    return ret;
  },
});

module.exports = mongoose.model("Student", studentSchema);