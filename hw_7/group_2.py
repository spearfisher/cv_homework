import os
import cv2
import numpy as np
import open3d as o3d


'''
Береза Денис
ДЗ 7.
Реалізувати умови завдання І рівня складності для кількості камер більше 2-х у
багатовидовій (мультіканальній) системі відеоспостереження.

Скрипт завантажує послідовність зображень, обчислює карту диспаратності для кожної пари сусідніх зображень.
Датасет сформований самостійно з камери телефону і USB камери(boba_fett).
Результати вийшли не дуже якісними, хоча для оригінальних зображень із лекції(алое) скрипт працює краще. 
Відслідковується залежність від якості датасету.


Що не вийшло реалізувати:
- спробував сформувати датасет (boba_fett), зображення брав коли камера рухалась довкола обєкту по радіусу
- отрмав дуже погані результати, певно алгоритм розраховує зміщення по горизонталі
'''

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

DATASETS = {
    '1': 'aloe',
    '2': 'airpods',
    '3': 'vog',
    '4': 'boba_fett',
}

window_size = 3
min_disp = 16
num_disp = 112 - min_disp


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def load_images(dataset_folder):
    dataset_path = os.path.join(DATASET_DIR, dataset_folder)
    image_files = sorted([
        f for f in os.listdir(dataset_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    images = []
    for name in image_files:
        path = os.path.join(dataset_path, name)
        img = cv2.pyrDown(cv2.imread(path))
        if img is None:
            print(f"  Warning: не вдалося завантажити {path}")
            continue
        images.append((name, img))
        print(f"  Завантажено: {name} → {img.shape[1]}×{img.shape[0]}")
    return images


def compute_pair(imgL, imgR):
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=16,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disp


def disp_to_pointcloud(disp, imgL):
    h, w = imgL.shape[:2]
    f = 0.8 * w
    Q = np.float32([
        [1,  0, 0, -0.5 * w],
        [0, -1, 0,  0.5 * h],
        [0,  0, 0,       -f],
        [0,  0, 1,        0]
    ])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    # Залишаємо тільки пікселі з достатньо великою диспаратністю
    # (низька диспаратність = далекий фон = конус)
    valid = disp > min_disp
    if valid.any():
        disp_thresh = np.percentile(disp[valid], 70)  # відкидаємо 70% найдальших
    else:
        disp_thresh = min_disp
    mask = (disp > disp_thresh) & np.isfinite(points).all(axis=2)
    pts = points[mask]
    cols = colors[mask]
    return pts, cols


def main(dataset_folder):
    print(f"\n=== Завантаження: {dataset_folder} ===")
    images = load_images(dataset_folder)
    if len(images) < 2:
        print("Потрібно мінімум 2 зображення")
        return

    all_points = []
    all_colors = []

    print(f"\n=== Обробка {len(images) - 1} пар ===")
    for i in range(len(images) - 1):
        nameL, imgL = images[i]
        nameR, imgR = images[i + 1]
        print(f"Пара: {nameL} ↔ {nameR}")

        disp = compute_pair(imgL, imgR)
        pts, cols = disp_to_pointcloud(disp, imgL)
        all_points.append(pts)
        all_colors.append(cols)
        print(f"  Точок: {len(pts)}")

        # Показуємо карту диспаратності як в оригіналі
        cv2.imshow(f'disparity {nameL}↔{nameR}', (disp - min_disp) / num_disp)

    # Зберігаємо окремі .ply для кожної пари і загальний
    merged_pts = np.vstack(all_points)
    merged_cols = np.vstack(all_colors)

    out_path = os.path.join(SCRIPT_DIR, f'out_{dataset_folder}.ply')
    write_ply(out_path, merged_pts, merged_cols)
    print(f"\nЗбережено: {out_path} ({len(merged_pts)} точок)")

    # open3d — злита хмара
    pcd = o3d.io.read_point_cloud(out_path)
    # Видаляємо статистичні викиди
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(pcd)

    cv2.waitKey(1)

    print("Відкриваємо 3D візуалізацію...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Multi-View 3D — {dataset_folder}",
        width=900, height=700
    )

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Оберіть датасет:')
    for key, name in DATASETS.items():
        print(f'{key} - {name}')

    choice = input('dataset: ').strip()
    if choice not in DATASETS:
        print(f"Невірний вибір: {choice}")
    else:
        main(DATASETS[choice])
