import os
import time

import cv2
from dotenv import load_dotenv

from frame_grabber import FrameGrabber
from zone_counter import ZoneEntryCounter

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")


def main():
    grabber = FrameGrabber(RTSP_URL).start()
    time.sleep(1.0)  # let the grabber pull in a first frame

    #cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    counter = ZoneEntryCounter(grabber).start()

    try:
        while True:
            frame = counter.read()

            if frame is None:
                ret, frame = grabber.read()
                if not ret:
                    continue

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print(f"Final zone entry count: {counter.entry_count}")
        print(f"Entry log: {counter.log_path}")
        counter.stop()
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()