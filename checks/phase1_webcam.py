# Phase 1, Steps 2-4: live webcam + detection + distance. Press 'q' to quit.

import sys
import os
import cv2


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision import Detector, analyze, draw


detector = Detector()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam. Try cv2.VideoCapture(1).")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, 480))
    result = analyze(detector, frame, do_ocr=False)
    frame = draw(frame, result["objects"])
    if result["quality"]["warnings"]:
        cv2.putText(frame, "WARN: " + ",".join(result["quality"]["warnings"]), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("VisionPath AI - Phase 1", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
