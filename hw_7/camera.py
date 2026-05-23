import os
import cv2

SAVE_DIR = "./dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera 0", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        path = os.path.join(SAVE_DIR, f"img_{img_counter}.jpg")
        cv2.imwrite(path, frame)
        print(f"Saved: {path}")
        img_counter += 1
    elif key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()



