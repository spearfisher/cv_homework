import os
import cv2
from ultralytics import YOLO


'''
Береза Денис
ДЗ 6.
Із впровадженням технологій штучних нейронних мереж розробити програмний скрипт для
розпізнавання автівок за їх типом у відеопотоці.

Скрипт реалізує детекцію автівок за їх типом(легкові, вантажівки) у відеопотоці.

Що цікаво, параметер conf вплває на дальність обєктів розпізнавання.
Чим він вищий, тим ближче до камери має бути обєкт, щоб його розпізнали.
'''


# Loads model. Let's use a small for performance.
model = YOLO("yolo26n.pt") 
# print(model.names)
classes = [2, 7] # 2 - car, 7 - truck

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
VIDEO_1 = os.path.join(DATASET_DIR, '00067cfb-5443fe39.mov')

cap = cv2.VideoCapture(VIDEO_1)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Відео завершилось")
        break

    results = model.predict(
        source=frame,
        classes=classes,
        conf=0.7,
        iou=0.45, # усуває зайві детекції, які перекриваються з іншими
        verbose=False
      )

    # Візуалізація
    annotated_frame = results[0].plot(
        line_width=2
        # labels=False, conf=False
    )
    cv2.imshow("YOLO Car Types", annotated_frame)

    c = cv2.waitKey(1)
    if c == 27 or c == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
