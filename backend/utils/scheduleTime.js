const INDIA_TIME_ZONE = "Asia/Kolkata";

function getIndiaDateTime(now = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: INDIA_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(now);

  const values = Object.fromEntries(
    parts
      .filter(({ type }) => type !== "literal")
      .map(({ type, value }) => [type, Number(value)])
  );

  return values;
}

function getIndiaDateString(now = new Date()) {
  const { year, month, day } = getIndiaDateTime(now);
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function parseTime(value) {
  const match = String(value || "")
    .trim()
    .match(/^(\d{1,2}):(\d{2})(?:\s*(AM|PM))?$/i);

  if (!match) {
    return null;
  }

  let hour = Number(match[1]);
  const minute = Number(match[2]);
  const meridiem = match[3]?.toUpperCase();

  if (minute > 59 || hour > (meridiem ? 12 : 23)) {
    return null;
  }

  if (meridiem) {
    if (hour === 12) {
      hour = 0;
    }
    if (meridiem === "PM") {
      hour += 12;
    }
  }

  return hour * 60 + minute;
}

function formatTime12(value) {
  const minutes = parseTime(value);
  if (minutes === null) {
    return "";
  }

  const hour24 = Math.floor(minutes / 60);
  const hour12 = hour24 % 12 || 12;
  const meridiem = hour24 >= 12 ? "PM" : "AM";
  return `${String(hour12).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")} ${meridiem}`;
}

function formatTime24(value) {
  const minutes = parseTime(value);
  if (minutes === null) {
    return "";
  }

  return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
}

function normalizeScheduleTime(startTime, endTime) {
  const normalizedStart = formatTime24(startTime);
  const normalizedEnd = formatTime24(endTime);
  if (!normalizedStart || !normalizedEnd) {
    return null;
  }

  return `${normalizedStart} - ${normalizedEnd}`;
}

function getScheduleTimeRange(schedule) {
  const [startTime, endTime] = String(schedule?.time || "").split(" - ");
  const startMinutes = parseTime(startTime);
  const endMinutes = parseTime(endTime);

  if (!schedule?.date || startMinutes === null || endMinutes === null) {
    return null;
  }

  const dateMatch = String(schedule.date).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!dateMatch) {
    return null;
  }

  const dateSerial =
    Date.UTC(Number(dateMatch[1]), Number(dateMatch[2]) - 1, Number(dateMatch[3])) /
    86400000;
  const startSerial = dateSerial * 1440 + startMinutes;
  const endSerial =
    dateSerial * 1440 + (endMinutes <= startMinutes ? endMinutes + 1440 : endMinutes);

  return { startSerial, endSerial };
}

function getScheduleTimeState(schedule, now = new Date()) {
  const range = getScheduleTimeRange(schedule);
  if (!range) {
    return { status: "invalid", startSerial: null, endSerial: null };
  }

  const indiaNow = getIndiaDateTime(now);
  const nowSerial =
    Date.UTC(indiaNow.year, indiaNow.month - 1, indiaNow.day) / 86400000 * 1440 +
    indiaNow.hour * 60 +
    indiaNow.minute;

  if (nowSerial < range.startSerial) {
    return { status: "upcoming", ...range };
  }
  if (nowSerial >= range.endSerial) {
    return { status: "expired", ...range };
  }
  return { status: "active", ...range };
}

module.exports = {
  INDIA_TIME_ZONE,
  formatTime12,
  formatTime24,
  getIndiaDateString,
  getScheduleTimeRange,
  getScheduleTimeState,
  normalizeScheduleTime,
};
