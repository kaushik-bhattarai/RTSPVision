import os
import time
import cv2
from dotenv import load_dotenv

from frame_grabber import FrameGrabber
from zone_counter import ZoneEntryCounter

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")

if not RTSP_URL:
    raise RuntimeError("RTSP_URL not found in .env")

def main():
    grabber = FrameGrabber(RTSP_URL).start()

    # Give the grabber time to fetch the first frame.
    time.sleep(1.0)

    counter = ZoneEntryCounter(grabber).start()

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera",cv2.WND_PROP_AUTOSIZE,cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1920, 1080)  #screen size

    try:
        while True:
            if counter.crashed:
                print("Zone counter worker thread has crashed - see traceback above. Stopping.")
                break

            frame = counter.read()

            # If no processed frame yet, display the raw frame.
            if frame is None:
                ret, frame = grabber.read()

                if not ret:
                    time.sleep(0.01)
                    continue

            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print(f"Final zone entry count: {counter.entry_count}")

        counter.stop()
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()