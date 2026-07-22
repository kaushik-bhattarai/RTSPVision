import os
import csv
import time
import threading
import traceback
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

import reid_tracker

#region points for the polygonal zone to monitor. 
REGION_POINTS = [(1035,570), (1088,620), (1186,0), (1100,0)]

MODEL_PATH = "yolov8n.pt"
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.5

CAPTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
ENTRY_LOG_PATH = os.path.join(CAPTURES_DIR, "entries_log.csv")


class ZoneEntryCounter:
    """
    Own thread. Runs YOLO human detection on the full frame, draws a box
    for every detected person, and checks whether ANY person's point
    currently falls inside the polygon.
    """

    def __init__(
        self,
        grabber,
        region_points=REGION_POINTS,
        model_path=MODEL_PATH,
        captures_dir=CAPTURES_DIR,
        log_path=ENTRY_LOG_PATH,
    ):
        self.grabber = grabber
        self.model = YOLO(model_path)
        self.region_np = np.array(region_points, dtype=np.int32)
        self.captures_dir = captures_dir
        self.log_path = log_path
        self.lock = threading.Lock()
        self.annotated_frame = None
        self.entry_count = 0
        self.exit_count = 0
        self._event_count = self._prepare_storage() # Start counting from the last event ID in the log
        self.flag = False

        # Internal only - used to detect the rising edge.
        self._zone_occupied_prev = False

        self.stopped = False
        self.crashed = False

        self.thread = threading.Thread(target=self._run, daemon=True)

    def _prepare_storage(self):
        os.makedirs(self.captures_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["event_id", "event_type", "timestamp", "image_path"])
            return 0
        
        last_event_id = 0
        with open(self.log_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    last_event_id = int(row[0])
        return last_event_id

    def start(self):
        self.thread.start()
        return self

    def _is_inside(self, point):
        return cv2.pointPolygonTest(self.region_np, point, False) >= 0

    def _save_event(self, frame, box_xyxy, event_type):
        """Crop the relevant person's box (or save the full frame if no
        box is available at the moment of the trigger), then log it."""
        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
        else:
            crop = frame

        self._event_count += 1
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"event_{self._event_count:04d}_{event_type}_{timestamp_str}.jpg"
        image_path = os.path.join(self.captures_dir, filename)

        cv2.imwrite(image_path, crop)

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self._event_count,
                    event_type,
                    now.isoformat(sep=" ", timespec="seconds"),
                    image_path,
                ]
            )

        if event_type == "enter":
            try:
                result = reid_tracker.process_entry(
                    crop, image_path=image_path, timestamp=now
                )
                if result["is_new"]:
                    print(f"[REID] New person recorded (ID {result['person_id']})")
                else:
                    last = result["last_seen_before"]
                    print(
                        f"[REID] Person {result['person_id']} recognized - "
                        f"last seen {last['timestamp']} ({last['image_path']}), "
                        f"visit #{result['visit_count']} today"
                    )
            except Exception:
                print("[REID] process_entry failed, continuing without it:")
                traceback.print_exc()

    def _run(self):
        try:
            self._update()
        except Exception:
            self.crashed = True
            print("[ZoneEntryCounter] worker thread crashed:")
            traceback.print_exc()

    def _update(self):
        while not self.stopped:
            ret, frame = self.grabber.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Keep an unmarked copy so saved crops don't contain the
            # overlay boxes/text drawn onto `frame` below.
            clean_frame = frame.copy()

            results = self.model(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                verbose=False,
            )[0]

            zone_occupied_this_frame = False
            triggering_box = None

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Box centroid (center point of the bounding box)
                point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                inside = self._is_inside(point)

                if inside and triggering_box is None:
                    triggering_box = (x1, y1, x2, y2)
                zone_occupied_this_frame = zone_occupied_this_frame or inside

                color = (0, 255, 0) if inside else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"person {conf:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                cv2.circle(frame, point, 4, color, -1)

            # crossing: same trigger every time (rising edge). Which label
            # it gets - enter or exit - just alternates with self.flag.
            if zone_occupied_this_frame and not self._zone_occupied_prev:
                self.flag = not self.flag
                if self.flag:
                    self.entry_count += 1
                    self._save_event(clean_frame, triggering_box, "enter")
                else:
                    self.exit_count += 1
                    self._save_event(clean_frame, triggering_box, "exit")

            self._zone_occupied_prev = zone_occupied_this_frame

            # Draw the zone polygon
            cv2.polylines(
                frame, [self.region_np], isClosed=True, color=(255, 0, 0), thickness=2
            )

            # Overlay counters
            cv2.putText(
                frame,
                f"Entries: {self.entry_count}  Exits: {self.exit_count}",
                (1280, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Flag: {self.flag}",
                (1280, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            with self.lock:
                self.annotated_frame = frame

    def read(self):
        with self.lock:
            if self.annotated_frame is None:
                return None
            return self.annotated_frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)