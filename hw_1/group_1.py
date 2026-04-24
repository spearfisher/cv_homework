import os
import cv2
import numpy as np

from shared_functions import loadImagesPaths, readImage


"""
Береза Денис
Група вимог №1.

Ідентифікація літаків на злітних смузах аеропорту.
Даний скрипт підвантажує датасет зображень злітниз смуг, обробляє їх і виводить результат підрахунку на екран.
Навігація між зображеннями здійснюється за допомогою клавіш WASD. Для виходу натисніть 'Q' або 'ESC'.
NOTE: Для навігації обовязково встановити англійську розкладку клавіатури.

Що вдалося реалізувати:
1. Препроцесинг зображень: розмиття, бінаризація, морфологічні операції для виділення контурів.
2. Фільтрація контурів за площею та співвідношенням площі контуру до площі його bounding box.
3. Використання SIFT для порівняння і фільтрації ROI з етелонним зображенням літака.

Покращення, які можна внести при наявності додаткового часу:
1. Більш гнучний підбір порогу біниразації. Автоматичний підбір масштабу
2. Заміна SIFT на інший алгоритм. SIFT часто давав хибні результати.
3. Підбір кращого шаблону літака для порівняння. Або використання декількох шаблонів для різних типів і ракурсів літаків.
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, 'dataset/airport/')

MIN_CONTOUR_AREA_RATIO = 0.0005
MAX_CONTOUR_AREA_RATIO = 0.8
MAX_BBOX_AREA_RATIO = 0.05
MEDIAN_THRESHOLD = 127

airplain_template = readImage(os.path.join(script_dir, 'dataset', 'airplane_3.png'), cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(airplain_template, None)
bf = cv2.BFMatcher()

def process_image(original):
    objectsCounter = 0
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    median = np.median(original)

    blur = cv2.GaussianBlur(gray, (7,7), 0)

    threshold = 205 if median > MEDIAN_THRESHOLD else 175
    _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    kernel_size = 20 if median > MEDIAN_THRESHOLD else 9
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)

    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = original.shape[0] * original.shape[1]
    min_contour_area = image_area * MIN_CONTOUR_AREA_RATIO
    max_contour_area = image_area * MAX_CONTOUR_AREA_RATIO
    max_bbox_area = image_area * MAX_BBOX_AREA_RATIO

    filtered_contours = []
    bboxes = []
    removed_contours = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        if bbox_area >= max_bbox_area:
            removed_contours.append(contour)
        elif min_contour_area <= contour_area <= max_contour_area and contour_area * 3 < bbox_area:
            filtered_contours.append(contour)
            bboxes.append((x, y, w, h))
        else:
            removed_contours.append(contour)

    result = original.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), 2)
    # cv2.drawContours(result, removed_contours, -1, (255, 0, 0), 2)

    for (x, y, w, h) in bboxes:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0: continue

        _, des_roi = sift.detectAndCompute(roi, None)
        if des_roi is not None:
            matches = bf.knnMatch(des_ref, des_roi, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.9 * n.distance:
                    good.append(m)

            if len(good) > 10:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, f"Matches: {len(good)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                objectsCounter += 1

    if objectsCounter > 0:
      cv2.putText(result, f"Detected Airplanes: {objectsCounter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result



if __name__ == '__main__':
    image_paths = loadImagesPaths(dataset_dir)
    total_images = len(image_paths)
    current_idx = 0

    print("Controls: 'D' - Next, 'A' - Prev, 'Q' or 'ESC' - Exit")
    win_name = 'Airplains Processor'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    image_changed = True
    while True:
        if image_changed:
            original = readImage(image_paths[current_idx])
            result = process_image(original)
            cv2.imshow(win_name, result)
            image_changed = False

        # --- Обробка клавіатури ---
        key = cv2.waitKey(300)
        key_byte = key & 0xFF

        if key in [27, ord('q')]:
            break
        elif key_byte in [ord('a'), ord('s')] or key in [81, 2424832, 84, 2621440]:
            current_idx = (current_idx + 1) % total_images
            image_changed = True
        elif key_byte in [ord('d'), ord('w')] or key in [83, 2555904, 82, 2490368]:
            current_idx = (current_idx - 1 + total_images) % total_images
            image_changed = True

    cv2.destroyAllWindows()
