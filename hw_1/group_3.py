import os
import cv2
import numpy as np

from shared_functions import readImage

"""
Береза Денис
Група вимог №3.

Підрахунок будівель КПІ на знімках з супутника.
Даний скрипт підвантажує два зображення КПІ з різних джерел (Google і Bing)
Для порівняння методів я застосовав різні методи для виявлення будівель на зображеннях.
Google - використовував пошук контурів після фільтрації Canny, а для Bing - колірну сегментацію в HSV-просторі.
"""

script_dir = os.path.dirname(os.path.abspath(__file__))

# Пошук контурів на зображенні та їх фільтрація за площею і співвідношенням сторін.
# Дає багато помилкових результатів, пропускає будівлі внизу зображення за рахунок іншої освітленості
def process_google_image(original):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 44, 166)

    # 3. Трішки розширимо лінії, щоб замкнути контури будинків
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1600:
            x, y, w, h = cv2.boundingRect(cnt)

            # Рахуємо площу прямокутника, щоб відсіяти "палиці" (дороги)
            if w > 0 and h > 0:
                aspect_ratio = max(w, h) / min(w, h)

                # Якщо об'єкт не занадто довгий (як дорога) — це будинок
                if aspect_ratio < 6:
                    building_count += 1
                    cv2.drawContours(original, [cnt], -1, (0, 0, 255), 2)
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(original, f"Google: {building_count} buildings found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(f"Знайдено прямокутних об'єктів: {building_count}")

    return building_count

# Метод, який дає кращі результати за рахунок підбору кольору будівель в HSV-просторі.
# Дуже залежить від кольорового простору і освітлення, тому не працює для знімків з іншими параметрами.
# Чутливий до доріг і тротуарів, які мають схожі кольори з будівлями.
def process_bing_image(original):
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # підібрані значення HSV
    lower_val = np.array([53, 6, 70])
    upper_val = np.array([179, 255, 255])
    mask_color = cv2.inRange(hsv, lower_val, upper_val)
    # підібрані значення сірого
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([179, 35, 240]) 
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    full_mask = cv2.bitwise_or(mask_color, mask_gray)

    # морфологія
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph_noise = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    morph_join = cv2.morphologyEx(morph_noise, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph_join, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1600:
            building_count += 1
            cv2.drawContours(original, [cnt], -1, (0, 0, 255), 2)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(original, f"Bing: {building_count} buildings found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(f"Знайдено ймовірних будівель: {building_count}")

    return building_count

if __name__ == '__main__':
    win_name = 'KPI Buildings Counter'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    image_changed = True
    while True:
        if image_changed:
            bing_img = readImage(os.path.join(script_dir, 'dataset/kpi/kpi_bing.png'))
            bing_count = process_bing_image(bing_img)

            google_img = readImage(os.path.join(script_dir, 'dataset/kpi/kpi_bing.png'))
            google_count = process_google_image(google_img)

            combined = np.hstack((bing_img, google_img))
            cv2.imshow(win_name, combined)
            image_changed = False

        key = cv2.waitKey(300)
        key_byte = key & 0xFF
        if key in [27, ord('q')]:
            break

    cv2.destroyAllWindows()
