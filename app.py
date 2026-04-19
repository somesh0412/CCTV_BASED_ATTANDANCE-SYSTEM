"""
app.py - Flask Web Application
"""
import os, csv
from flask import Flask, render_template, Response, jsonify, send_file, abort
import recognize

app = Flask(__name__)
ATTENDANCE_CSV = "attendance.csv"
FACES_DIR      = "faces"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(recognize.generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    return jsonify({
        "active":  recognize.get_camera_active(),
        "message": recognize.get_status()
    })

@app.route("/api/attendance")
def api_attendance():
    # ── Primary source: in-memory log from recognition thread (instant) ──
    live = recognize.get_attendance()   # list of {Name, Date, Time}

    # ── Fallback: also read CSV for records from previous sessions ──────
    csv_records = []
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                # Skip blank rows that old buggy code wrote
                if row.get("Name","").strip():
                    csv_records.append(row)

    # Merge: live takes priority; add CSV records not already in live
    live_names = {r["Name"] for r in live}
    for row in csv_records:
        if row["Name"] not in live_names:
            live.append(row)

    return jsonify(live)   # newest first (live list is already newest-first)


@app.route("/api/registered")
def api_registered():
    """Return all registered persons with present/absent status."""
    # Who is present right now (in-memory, this session)
    present_names = {r["Name"] for r in recognize.get_attendance()}

    persons = []
    if os.path.exists(FACES_DIR):
        for name in sorted(os.listdir(FACES_DIR)):
            p = os.path.join(FACES_DIR, name)
            if os.path.isdir(p):
                img_count = len([f for f in os.listdir(p)
                                  if f.lower().endswith(('.jpg','.jpeg','.png'))])
                persons.append({
                    "name":    name,
                    "images":  img_count,
                    "present": name in present_names   # ← NEW: live status
                })
    return jsonify(persons)


@app.route("/download/attendance")
def download_attendance():
    if not os.path.exists(ATTENDANCE_CSV):
        abort(404)
    return send_file(os.path.abspath(ATTENDANCE_CSV), mimetype="text/csv",
                     as_attachment=True, download_name="attendance.csv")


@app.route("/api/start", methods=["POST"])
def api_start():
    if not recognize.get_camera_active():
        recognize.start()
        return jsonify({"ok": True, "message": "Recognition started."})
    return jsonify({"ok": False, "message": "Already running."})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    recognize.stop()
    return jsonify({"ok": True, "message": "Stopped."})


if __name__ == "__main__":
    recognize.start()
    print("\n" + "="*55)
    print("  CCTV Attendance System")
    print("  → Open: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
