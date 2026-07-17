import os
import time
import threading

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")

RTSP_URL = (
    f"rtsp://{USERNAME}:{PASSWORD}"
    f"@192.168.110.4:554/Streaming/Channels/102"
)

PERSON_CLASS_ID = 0  # "person" in the default COCO-trained YOLO models
CONF_THRESHOLD = 0.5


class FrameGrabber:
    """
    Runs in its own thread. Continuously reads frames from the RTSP
    stream and always keeps only the MOST RECENT frame.
    """

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream")

        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False

        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # Stream hiccup - back off briefly and retry instead of
                # spinning a hot loop or killing the thread outright.
                time.sleep(0.05)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


class Detector:
    """
    Runs in its own thread. Pulls the latest frame from the grabber,
    runs YOLO inference, and stores the latest annotated frame +
    detection results. Like the grabber, it always works on the
    newest available frame rather than a queue, so inference speed
    doesn't create a backlog.
    """

    def __init__(self, grabber, model_path="yolov8n.pt"):
        self.grabber = grabber
        self.model = YOLO(model_path)

        self.lock = threading.Lock()
        self.annotated_frame = None
        self.detections = []
        self.stopped = False

        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.grabber.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            results = self.model(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                verbose=False,
            )[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {conf:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            with self.lock:
                self.annotated_frame = frame
                self.detections = detections

    def read(self):
        with self.lock:
            if self.annotated_frame is None:
                return None, []
            return self.annotated_frame.copy(), list(self.detections)

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)


def main():
    grabber = FrameGrabber(RTSP_URL).start()

    # Give the grabber a moment to pull in the first frame before the detector starts asking for one.
    time.sleep(1.0)

    detector = Detector(grabber, model_path="yolov8n.pt").start()

    try:
        while True:
            frame, detections = detector.read()

            if frame is None:
                # Detector hasn't produced a frame yet - fall back to
                # the raw feed so the window isn't just blank.
                ret, frame = grabber.read()
                if not ret:
                    continue

            cv2.imshow("Camera - Human Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.stop()
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()