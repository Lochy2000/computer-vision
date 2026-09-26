"""Local webcam face recognition and presence reporting.

Faces must be enrolled explicitly. Recognition and storage remain local.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "cctv_data"
DETECTOR_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS / "face_recognition_sface_2021dec.onnx"
DATABASE_FILE = DATA / "faces.json"
EVENT_FILE = DATA / "events.csv"
REPORT_FILE = DATA / "report.html"
MATCH_THRESHOLD = 0.363


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9 _-]", "", value).strip()
    if not value:
        raise ValueError("Name must contain a letter or number")
    return value[:60]


def ensure_models() -> None:
    missing = [path for path in (DETECTOR_MODEL, RECOGNIZER_MODEL) if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing model(s): {names}. Run .\\download_face_models.ps1 first."
        )


def load_database() -> dict[str, list[list[float]]]:
    if not DATABASE_FILE.exists():
        return {}
    with DATABASE_FILE.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {name: samples for name, samples in raw.get("people", {}).items()}


def save_database(people: dict[str, list[list[float]]]) -> None:
    DATA.mkdir(exist_ok=True)
    payload = {"version": 1, "people": people}
    DATABASE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FaceEngine:
    def __init__(self) -> None:
        ensure_models()
        self.detector = cv.FaceDetectorYN.create(
            str(DETECTOR_MODEL), "", (320, 320), 0.8, 0.3, 5000
        )
        self.recognizer = cv.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces

    def embedding(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feature)
        return feature / norm if norm else feature


def identify(feature: np.ndarray, people: dict[str, list[list[float]]]) -> tuple[str, float]:
    best_name, best_score = "Unknown", -1.0
    for name, samples in people.items():
        for sample in samples:
            score = float(np.dot(feature, np.asarray(sample, dtype=np.float32)))
            if score > best_score:
                best_name, best_score = name, score
    if best_score < MATCH_THRESHOLD:
        return "Unknown", max(best_score, 0.0)
    return best_name, best_score


def draw_face(frame: np.ndarray, face: np.ndarray, label: str, score: float) -> None:
    x, y, width, height = face[:4].astype(int)
    known = label != "Unknown"
    colour = (40, 200, 40) if known else (0, 165, 255)
    cv.rectangle(frame, (x, y), (x + width, y + height), colour, 2)
    text = f"{label} {score:.2f}" if known else label
    cv.putText(frame, text, (x, max(24, y - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


def open_camera(camera: int) -> cv.VideoCapture:
    capture = cv.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera}")
    return capture


def enroll(name: str, camera: int, samples_needed: int) -> None:
    name = safe_name(name)
    engine = FaceEngine()
    capture = open_camera(camera)
    samples: list[list[float]] = []
    print("Look at the camera and slowly turn your head. Press Q to cancel.")
    try:
        while len(samples) < samples_needed:
            ok, frame = capture.read()
            if not ok:
                break
            faces = engine.detect(frame)
            message = "Show exactly one face"
            if len(faces) == 1:
                face = faces[0]
                draw_face(frame, face, f"Sample {len(samples) + 1}/{samples_needed}", 0)
                message = "Hold still"
                if int(time.monotonic() * 2) % 2 == 0:
                    samples.append(engine.embedding(frame, face).tolist())
                    time.sleep(0.25)
            cv.putText(frame, message, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.imshow("Enrol face", frame)
            if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                samples.clear()
                break
    finally:
        capture.release()
        cv.destroyAllWindows()

    if not samples:
        print("Enrolment cancelled; nothing was saved.")
        return
    people = load_database()
    people[name] = samples
    save_database(people)
    print(f"Enrolled {name} with {len(samples)} face samples.")


@dataclass
class Presence:
    first_seen: datetime
    last_seen: datetime
    best_score: float


def append_event(name: str, presence: Presence) -> None:
    DATA.mkdir(exist_ok=True)
    new_file = not EVENT_FILE.exists()
    with EVENT_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(("name", "first_seen", "last_seen", "duration_seconds", "best_match"))
        duration = (presence.last_seen - presence.first_seen).total_seconds()
        writer.writerow((
            name,
            presence.first_seen.isoformat(timespec="seconds"),
            presence.last_seen.isoformat(timespec="seconds"),
            f"{duration:.1f}",
            f"{presence.best_score:.3f}",
        ))


def run(camera: int, absence_seconds: float) -> None:
    engine = FaceEngine()
    people = load_database()
    if not people:
        print("No enrolled people yet. Use: python face_cctv.py enroll --name NAME")
    capture = open_camera(camera)
    active: dict[str, Presence] = {}
    print("Running locally. Press Q or Escape to stop.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = datetime.now()
            seen_now: set[str] = set()
            for face in engine.detect(frame):
                feature = engine.embedding(frame, face)
                name, score = identify(feature, people)
                draw_face(frame, face, name, score)
                if name != "Unknown":
                    seen_now.add(name)
                    if name not in active:
                        active[name] = Presence(now, now, score)
                    else:
                        active[name].last_seen = now
                        active[name].best_score = max(active[name].best_score, score)

            expired = [
                name for name, item in active.items()
                if name not in seen_now and (now - item.last_seen).total_seconds() >= absence_seconds
            ]
            for name in expired:
                append_event(name, active.pop(name))

            cv.putText(frame, f"Known people present: {len(active)}", (10, 28),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.imshow("Local Face CCTV", frame)
            if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        for name, presence in active.items():
            append_event(name, presence)
        capture.release()
        cv.destroyAllWindows()


def make_report() -> None:
    rows: list[dict[str, str]] = []
    if EVENT_FILE.exists():
        with EVENT_FILE.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    totals: dict[str, float] = {}
    for row in rows:
        totals[row["name"]] = totals.get(row["name"], 0) + float(row["duration_seconds"])
    summary = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{sum(r['name'] == name for r in rows)}</td>"
        f"<td>{seconds:.1f}</td></tr>" for name, seconds in sorted(totals.items())
    ) or "<tr><td colspan='3'>No recorded visits</td></tr>"
    detail = "".join(
        "<tr>" + "".join(f"<td>{html.escape(row[key])}</td>" for key in
        ("name", "first_seen", "last_seen", "duration_seconds", "best_match")) + "</tr>"
        for row in reversed(rows)
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CCTV activity report</title><style>
body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin-bottom:35px}}th,td{{padding:10px;border:1px solid #ccd1d1;text-align:left}}
th{{background:#eaf2f8}}small{{color:#566573}}</style></head><body>
<h1>CCTV activity report</h1><small>Generated {datetime.now().isoformat(timespec='seconds')}</small>
<h2>Summary</h2><table><tr><th>Person</th><th>Visits</th><th>Total seconds</th></tr>{summary}</table>
<h2>Presence log</h2><table><tr><th>Person</th><th>First seen</th><th>Last seen</th><th>Seconds</th><th>Best match</th></tr>{detail}</table>
</body></html>"""
    DATA.mkdir(exist_ok=True)
    REPORT_FILE.write_text(document, encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")


def remove(name: str) -> None:
    people = load_database()
    if name not in people:
        raise KeyError(f"No enrolled person named {name}")
    del people[name]
    save_database(people)
    print(f"Removed {name} from the local face database.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local face recognition and presence reporting")
    sub = parser.add_subparsers(dest="command", required=True)
    enrol = sub.add_parser("enroll", help="Capture face samples for one consenting person")
    enrol.add_argument("--name", required=True)
    enrol.add_argument("--camera", type=int, default=0)
    enrol.add_argument("--samples", type=int, default=10)
    watch = sub.add_parser("run", help="Recognise enrolled faces from the webcam")
    watch.add_argument("--camera", type=int, default=0)
    watch.add_argument("--absence-seconds", type=float, default=5.0)
    sub.add_parser("report", help="Generate an HTML report from the event log")
    delete = sub.add_parser("remove", help="Delete an enrolled identity")
    delete.add_argument("--name", required=True)
    sub.add_parser("list", help="List locally enrolled identities")
    args = parser.parse_args()

    if args.command == "enroll":
        enroll(args.name, args.camera, max(3, args.samples))
    elif args.command == "run":
        run(args.camera, args.absence_seconds)
    elif args.command == "report":
        make_report()
    elif args.command == "remove":
        remove(args.name)
    elif args.command == "list":
        people = load_database()
        print("\n".join(sorted(people)) if people else "No enrolled people")


if __name__ == "__main__":
    main()
