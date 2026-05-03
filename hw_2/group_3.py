import os
import cv2
import numpy as np

from shared_functions import readImage

"""
Береза Денис
Фільтрація та покращення якості цифрових зображень
Група вимог №3.

Підрахунок будівель КПІ на знімках з супутника.
Даний скрипт підвантажує два зображення КПІ з різних джерел (Google і Bing)
Цікаві результати дає білатеральний фільтр із великим діаметром розмиття(Google)
Для зображення з Bing використав білатеральний фільтр і вирівнювання гістограми, але більше шумів
"""

script_dir = os.path.dirname(os.path.abspath(__file__))

def process_bing_bilateral(original):
    filtered_img = cv2.bilateralFilter(original, d=17, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    edges = cv2.Canny(equ, 50, 500)

    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1900:
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



def process_google_bilateral(original):
    # Цікаві результати дає білатеральний фільтр із великим діаметром розмиття
    filtered_img = cv2.bilateralFilter(original, d=17, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 300)

    # Розширимо лінії, щоб замкнути контури будинків
    # Цього разу підвищими ядро, адже білатеральний фільтр досить сильно розмиває зображення
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1900:
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

if __name__ == '__main__':
    win_name = 'KPI Buildings Counter'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    image_changed = True
    while True:
        if image_changed:
            bing_img = readImage(os.path.join(script_dir, 'dataset/kpi/kpi_bing.png'))
            bing_count = process_bing_bilateral(bing_img)

            google_img = readImage(os.path.join(script_dir, 'dataset/kpi/kpi_google.png'))
            google_count = process_google_bilateral(google_img)

            combined = np.hstack((bing_img, google_img))
            cv2.imshow(win_name, combined)
            image_changed = False

        key = cv2.waitKey(300)
        key_byte = key & 0xFF
        if key in [27, ord('q')]:
            break

    cv2.destroyAllWindows()
