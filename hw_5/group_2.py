import cv2
import os

'''
Береза Денис
ДЗ 5.
Технології ідентифікації в Computer Vision
Група вимог №2.

Скрипт реалізує комбіновану індетифікацію людських силуетів декількома каскадами Хаара (повний силует, нижня частина).
Верхня частина індетифікується погано і дає багато хибних спрацювань, тому в фінальний код не потрапила.

Для зменшення хибних спрацювань використовується:
- задання обмежень на розмір тіла людини (висота, ширина);
- розмиття зображення;

Каскади досить швидко опрацьовують відео, можуть бути використані на малопотужних пристроях.

Що не вийшло:
При зйомці зверху класифікатор не працював, адже натренований на зйомці в горизонтальній площині.
Для такої орієнтації потрібне застосування класифікаторів натренованих на зйомці згори.
Автоматичний підбір класифікатора можна реалізувати від показників альтиметру дрону, куту нахилу(YAW).

Покращення на майбутнє:
Спробувати натренувати власні каскади на зйомці зверху;
Проте потрібна дуже велика навчальна вибірка(1000-5000 позитивних зображень, 3000-10000 негативних зображень);
'''


# Обмеження розміру тіла для кращого розпізнавання
# В реальній системі можна використати ROI від оператора для параметрів maxSize і minSize
BODY_HEIGHT = 80 # px
BODY_WIDTH = BODY_HEIGHT // 2 # px
LEGS_HEIGHT = BODY_HEIGHT // 2 # px

FRAME_RESIZE_FACTOR = 0.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

VIDEO_1 = os.path.join(DATASET_DIR, 'IMG_5147.MP4')
VIDEO_2 = os.path.join(DATASET_DIR, 'IMG_5286.MP4')


BODY_CASCADE = cv2.CascadeClassifier('haarcascade_fullbody.xml')
if BODY_CASCADE.empty():
	raise IOError('Unable to load the cascade classifier xml file')

LEGS_CASCADE = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
if LEGS_CASCADE.empty():
  raise IOError('Unable to load the cascade classifier xml file')

def process_video(file_path):
  cap = cv2.VideoCapture(file_path)

  while True:
      ret, frame = cap.read()
      if not ret or frame is None:
          break
      frame = cv2.resize(frame, None, 
              fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR, 
              interpolation=cv2.INTER_AREA)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (3, 3), 0)

      full_body_rects = BODY_CASCADE.detectMultiScale(
          blur,
          scaleFactor=1.1,
          minNeighbors=3,
          minSize=(LEGS_HEIGHT, LEGS_HEIGHT),
          flags=cv2.CASCADE_SCALE_IMAGE
      )

      for (xb,yb,wb,hb) in full_body_rects:
          full_body_roi = gray[yb:yb+hb, xb:xb+wb]
          legs = LEGS_CASCADE.detectMultiScale(
              full_body_roi,
              scaleFactor=1.1,
              minNeighbors=3,
              minSize=(LEGS_HEIGHT, LEGS_HEIGHT)
          )

          for (xl,yl,wl,hl) in legs:
              cv2.rectangle(frame, (xl+xb,yl+yb), (xl+xb+wl,yl+yb+hl), (255,0,0), 3)
              cv2.rectangle(frame, (xb,yb), (xb+wb,yb+hb), (0,255,0), 3)

      cv2.imshow('Body Detector', frame)

      c = cv2.waitKey(1)
      if c == 27 or c == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()


def run_demo():
	for video_path in [VIDEO_1, VIDEO_2]:
		process_video(video_path)


if __name__ == '__main__':
	print('Оберіть режим:')
	print('1 - Demo')
	print('2 - First video')
	print('3 - Second video')

	mode = int(input('mode: '))

	if mode == 1:
		run_demo()
	if mode == 2:
		process_video(VIDEO_1)
	if mode == 3:
		process_video(VIDEO_2)
