import os
import cv2
import torch
from ultralytics import YOLO

# Автодетект: MPS (Apple), CUDA (NVIDIA), або CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


'''
Береза Денис
ДЗ 6.
Детекція людей у відеопотоці зі зброєю. Спочатку детектуємо людей, потім в межах розширеного бокса шукаємо зброю.
Скрипт трохи повільно працює за рахунок двох моделей. Друга модель тренована на 1024 тому досить важка.

Модель для зброї досить заслабка, тому довелося знизити conf до 0.35, щоб щось знаходило.
Але це швидше ніж збирати, анотувати і тренувати датасет із зброєю.
'''

# Stage 1: людина
person_model = YOLO("yolo26n.pt")
# Stage 2: зброя
# https://huggingface.co/HaiderKhan6410/weapon-yolo26x
weapon_model = YOLO("weapons_best.pt")

PAD = 40  # ореол навколо бокса людини в пікселях

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
VIDEO_1 = os.path.join(DATASET_DIR, 'evaluation.mp4')

cap = cv2.VideoCapture(VIDEO_1)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Відео завершилось")
        break

    h, w = frame.shape[:2]

    # Stage 1: детектуємо людей
    person_results = person_model.predict(
        source=frame,
        classes=[0],
        conf=0.5,
        iou=0.45,
        device=DEVICE,
        verbose=False
    )

    annotated_frame = person_results[0].plot(line_width=1)

    person_boxes = person_results[0].boxes
    if person_boxes is not None:
        for box in person_boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = box

            # Розширюємо бокс із ореолом, не виходячи за межі кадру
            cx1 = max(0, int(x1) - PAD)
            cy1 = max(0, int(y1) - PAD)
            cx2 = min(w, int(x2) + PAD)
            cy2 = min(h, int(y2) + PAD)

            crop = frame[cy1:cy2, cx1:cx2]

            # Stage 2: детектуємо зброю в кропі
            weapon_results = weapon_model.predict(
                source=crop,
                conf=0.35,
                device=DEVICE,
                verbose=False
            )

            weapon_boxes = weapon_results[0].boxes
            if weapon_boxes is not None and len(weapon_boxes):
                # Знайдено зброю — червона рамка навколо людини
                cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)

                weapon_cls_id = int(weapon_boxes.cls[0].item())
                cv2.putText(annotated_frame, "ARMED DETECTED",
                            (cx1, cy1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

    cv2.imshow("YOLO Human + Weapon Detection", annotated_frame)

    c = cv2.waitKey(1)
    if c == 27 or c == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
