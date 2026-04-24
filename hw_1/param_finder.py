import cv2
import os
import sys
import glob
import numpy as np

from shared_functions import loadImagesPaths, readImage

script_dir = os.path.dirname(os.path.abspath(__file__))

TARGETS = {
    'kpi': {
        'dataset': os.path.join(script_dir, 'dataset/kpi/'),
        'morph': cv2.MORPH_CROSS,
        'blur': (3, 3),
    },
    'airport': {
        'dataset': os.path.join(script_dir, 'dataset/airport/'),
        'morph': cv2.MORPH_ELLIPSE,
        'blur': (5, 5),
    }
}
target = sys.argv[1] if len(sys.argv) > 1 else 'kpi'
if target not in TARGETS:
    raise ValueError(f"Unknown target: {target}. Use one of: {', '.join(TARGETS)}")
TARGET = TARGETS[target]


current_idx = 0
need_update = True

def update(val=None):
    global need_update
    need_update = True

if __name__ == '__main__':
    try:
        image_paths = loadImagesPaths(TARGET['dataset'])
    except FileNotFoundError as e:
        print(e)
        exit()

    total_images = len(image_paths)

    print("Controls: 'D' - Next, 'A' - Prev, 'Q' or 'ESC' - Exit")
    win_name = 'Image Processor'

    # Вікно з трекбарами для регулювання параметрів перетворення і фільтрації
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Blur', win_name, 1, 50, update)
    cv2.createTrackbar('Threshold', win_name, 190, 255, update)
    cv2.createTrackbar('Canny Low', win_name, 50, 500, update)
    cv2.createTrackbar('Canny High', win_name, 150, 500, update)


    while True:
        if need_update:
            blur_val = cv2.getTrackbarPos('Blur', win_name)
            k_size = blur_val if blur_val % 2 != 0 else blur_val + 1
            # k_size = blur_val

            t_val = cv2.getTrackbarPos('Threshold', win_name)
            c_low = cv2.getTrackbarPos('Canny Low', win_name)
            c_high = cv2.getTrackbarPos('Canny High', win_name)

            # Завантаження та обробка
            original = readImage(image_paths[current_idx])
            median = np.median(original)
            print(f"Median pixel value: {median}")

            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (k_size,k_size), 0)

            edges = cv2.Canny(blur, c_low, c_high)

            # kernel = np.ones((3,3), np.uint8)
            # dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # 4. Шукаємо контури
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # _, binary = cv2.threshold(blur, t_val, 255, cv2.THRESH_BINARY)

            # m_kernel = cv2.getStructuringElement(TARGET['morph'], (k_size, k_size)) # MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS
            # binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, m_kernel)
            # binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, m_kernel)
            # 3. Пошук контурів
            # contours, hierarchy = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(original, contours, -1, (0, 0, 255), 2)

            cv2.imshow(win_name, original)
            need_update = False

        # --- Обробка клавіатури ---
        key = cv2.waitKey(300)
        key_byte = key & 0xFF

        if key in [27, ord('q')]: # Вихід
            break
        elif key_byte in [ord('a'), ord('s')] or key in [81, 2424832, 84, 2621440]: # Вперед
            current_idx = (current_idx + 1) % total_images
            update()
        elif key_byte in [ord('d'), ord('w')] or key in [83, 2555904, 82, 2490368]:
            current_idx = (current_idx - 1 + total_images) % total_images
            update()

    cv2.destroyAllWindows()
