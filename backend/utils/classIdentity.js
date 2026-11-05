function normalizeValue(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replace(/\s+/g, " ");
}

function parseClassDivision(value) {
  const normalized = normalizeValue(value);
  const tokens = normalized.replace(/[^A-Z0-9]+/g, " ").trim().split(/\s+/).filter(Boolean);
  const year = tokens.find((token) => ["FY", "SY", "TY", "FINALYEAR"].includes(token)) || null;
  const division = [...tokens].reverse().find((token) => /^[A-Z]$/.test(token)) || null;
  const program = tokens
    .filter((token) => token !== year && token !== division)
    .join(" ") || null;

  return {
    raw: normalized,
    year,
    division,
    program,
  };
}

function getStudentClass(student, account) {
  const classDivision = account?.classDivision || student?.className;
  return {
    department: normalizeValue(account?.department || student?.department),
    ...parseClassDivision(classDivision),
  };
}

function getScheduleClass(schedule) {
  const parsed = parseClassDivision(schedule?.classDivision);
  return {
    department: normalizeValue(schedule?.department),
    year: normalizeValue(schedule?.academicYear) || parsed.year,
    division: normalizeValue(schedule?.division) || parsed.division,
    program: normalizeValue(schedule?.program) || parsed.program,
    raw: parsed.raw,
  };
}

function isStudentEligibleForSchedule(studentClass, schedule) {
  const expected = getScheduleClass(schedule);
  const mismatches = [];

  if (expected.department && studentClass.department !== expected.department) {
    mismatches.push("department");
  }
  if (expected.year && studentClass.year !== expected.year) {
    mismatches.push("academic year");
  }
  if (expected.division && studentClass.division !== expected.division) {
    mismatches.push("division");
  }
  if (expected.program && studentClass.program && studentClass.program !== expected.program) {
    mismatches.push("program");
  }

  return {
    eligible: mismatches.length === 0,
    mismatches,
    expected,
  };
}

module.exports = {
  normalizeValue,
  parseClassDivision,
  getStudentClass,
  getScheduleClass,
  isStudentEligibleForSchedule,
};
