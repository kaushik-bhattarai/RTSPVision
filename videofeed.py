import cv2

cap = cv2.VideoCapture("rtsp://localhost:8554/test")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("RTSP", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()