

import cv2
import numpy as np
import os
import shutil

#*
# 
# 
'''
ГРУПА ВИМОГ 2.
Скрипт має два режими роботи:
1. Меню(1, 2, 3) Кластеризація різних груп зображень річки/дороги/міста кластеризуються окремо від полів/лісів. 
  - Найкращі результати кластеризації були для датасетів річки та міста, характерні ознаки були більш виражені. Меню(1, 3)
  - Дороги та поля мають схожі ознаки. Важко виділити ознаки, які б були стабільними для цієї групи знімків Меню(2)
2. Меню(4) Відділення маски доріг від фону за допомогою K-means кластеризації.
 - Для одного знімку можна добре відділити фон від штучно створених обєктів. При К=2, К=3(Збільшення К веде до погіршення результату)
   В подальшому можна виокремити дорогу серед полів та будівель іншими ознаками(контури, детектор ліній).

 Для знімків краще показав себе діапазон HSV насиченість 0-255, яскравість 0-255. RGB - менш працював стабільно

Що не вдалось реалізувати:
Повторне застосування K-means для виділення дороги і забудови на одному знімку.
Ідея була в тому, щоб спочатку відсіяти фон і штучні обєкти за допомого. К=2
Потім повторно застосувати K-means до маски пікселів, щоб виділити обєкти серед них. Проте це також не дало стабільного результату, в коді не представлено.
'''

# --- НАЛАШТУВАННЯ ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH_RIVER = os.path.join(SCRIPT_DIR, "dataset", "fields_and_river")
SOURCE_PATH_ROADS = os.path.join(SCRIPT_DIR, "dataset", "fields_and_roads")
SOURCE_PATH_CITIES = os.path.join(SCRIPT_DIR, "dataset", "fields_and_cities")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "sorted_clusters")
K_CLUSTERS = 2


def process_image_groups(source_path):
    output_path = os.path.join(OUTPUT_PATH, os.path.basename(os.path.normpath(source_path)))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    features_list = []
    filenames = []

    print("Аналіз групи зображень...")
    print("source_path:", source_path)
    image_files = sorted([file for file in os.listdir(source_path) if file.lower().endswith((".png", ".jpg", ".jpeg"))])
    for file in image_files:
        full_path = os.path.join(source_path, file)

        print(f"Обробка: {full_path}")
        img = cv2.imread(full_path)

        if img is None: continue

        r, g, b = cv2.split(img.astype(float))
        exg = np.mean(2*g - r - b)
        
        # У дороги насиченість кольору нижча
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_sat = cv2.mean(hsv)[1]
        # Текстурність
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        # Середня яскравість
        avg_val = cv2.mean(hsv)[2]
        feature_vector = [exg, avg_sat, std_dev, avg_val]

        features_list.append(feature_vector)
        filenames.append(file)

    print("КЛАСТЕРИЗАЦІЯ...")
    data = np.array(features_list, dtype=np.float32)
    data_norm = cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data_norm, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    for i in range(K_CLUSTERS):
        cluster_dir = os.path.join(output_path, f"Cluster_{i}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

    for idx, label in enumerate(labels.flatten()):
        src = os.path.join(source_path, filenames[idx])
        dst = os.path.join(output_path, f"Cluster_{label}", filenames[idx])
        shutil.copy(src, dst)


def build_road_mask(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    data = blur.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 1-2 ітерацій виявилось достатньо для виділення дороги, більше ітерацій працює довше без покращення результату
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

    bright_cluster = np.argmax(np.mean(centers, axis=1))
    mask_bright = (labels.reshape(img.shape[:2]) == bright_cluster).astype(np.uint8) * 255

    return mask_bright


def show_road_masks(source_path):
    image_files = sorted([file for file in os.listdir(source_path) if file.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not image_files:
        print(f"No images found in: {source_path}")
        return

    print("Controls: 'q' or 'esc' - next image / exit")
    window_name = 'Roads Preview'
    max_preview_width = 1280
    max_preview_height = 720

    for file_name in image_files:
        full_path = os.path.join(source_path, file_name)
        img = cv2.imread(full_path)
        if img is None:
            continue

        mask_bright = build_road_mask(img)
        highlighted = np.zeros_like(img)
        highlighted[mask_bright > 0] = img[mask_bright > 0]

        preview = np.hstack((img, highlighted))
        preview_height, preview_width = preview.shape[:2]
        scale = min(max_preview_width / preview_width, max_preview_height / preview_height, 1.0)
        if scale < 1.0:
            preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        cv2.putText(preview, 'Original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(preview, 'Highlighted', (preview.shape[1] // 2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(preview, file_name, (20, preview.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(window_name, preview)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            continue

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Оберіть режим:')
    print('1 - Річки та поля')
    print('2 - Поля та дороги')
    print('3 - Поля та населені пункти')
    print('4 - Відділення маски доріг від фону')

    mode = int(input('mode: '))
    if mode == 1:
        process_image_groups(SOURCE_PATH_RIVER)
    if mode == 2:
        process_image_groups(SOURCE_PATH_ROADS)
    if mode == 3:
        process_image_groups(SOURCE_PATH_CITIES)
    if mode == 4:
        show_road_masks(SOURCE_PATH_ROADS)
