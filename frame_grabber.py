import time
import threading
import cv2


class FrameGrabber:
    """
    Own thread. Continuously reads frames from the RTSP stream and
    always keeps only the MOST RECENT one, so a slow consumer never
    causes a growing backlog/lag.
    """

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

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