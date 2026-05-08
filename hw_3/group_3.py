import os
import cv2
import numpy as np

"""
Береза Денис
ДЗ 3.
Технології порівняння цифрових зображень та Object Tracking
Група вимог №2.

Скрипт реалізує алгоритми відстеження об'єкта на основі MeanShift та пошуку шаблону(matchTemplate).
Для покращення трекера MeanShift використовується:
- тривимірна гістограма в HSV-просторі за каналами [H, S, V] замість одноканальної моделі;
- аналіз каналу яскравості V в ROI для вибору маски побудови гістограми.

Трекера matchTemplate реалізує пошук по частині кадру відносно попереднього положення об'єкта.

В залежності від вибору ROI результат може відрізнятися для обох методів.
Чим більше ROI, тим гірше трекер справляється з відстеженням, може перестрибувати на інші автомобілі
Як варіант покращення можна реалізувати відділення фону від обєкту на етапі виділення ROI
"""


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

VIDEO_1 = os.path.join(DATASET_DIR, 'output_1.mp4')
VIDEO_2 = os.path.join(DATASET_DIR, 'output_2.mp4')

# Підібрані ROI для демонстраційних відео
DEMO_ROIS = {
	'output_1.mp4': [886, 466, 142, 70],
	'output_2.mp4': [1100, 556, 137, 79],
}



# Аналізує гістограму V-каналу вибраного ROI,
# визначає домінуючі піки та створює маску для виділення об'єктів з відповідною яскравістю.
def build_adaptive_roi_histogram(hsv_roi):
	v_channel = hsv_roi[:, :, 2]
	low_v = int(np.percentile(v_channel, 10))
	high_v = int(np.percentile(v_channel, 90))
	v_hist = cv2.calcHist([v_channel], [0], None, [32], [0, 256]).flatten()
	peak_threshold = float(v_hist.max()) * 0.2
	peak_bins = []

	for i in range(len(v_hist)):
		left = v_hist[i - 1] if i > 0 else -1
		right = v_hist[i + 1] if i < len(v_hist) - 1 else -1
		if v_hist[i] >= left and v_hist[i] >= right and v_hist[i] >= peak_threshold:
			if not peak_bins or i - peak_bins[-1] >= 4:
				peak_bins.append(i)
			elif v_hist[i] > v_hist[peak_bins[-1]]:
				peak_bins[-1] = i

	has_dark_peak = any(bin_idx <= 11 for bin_idx in peak_bins)
	has_light_peak = any(bin_idx >= 20 for bin_idx in peak_bins)
	dominant_peak = int(np.argmax(v_hist))

	if has_dark_peak and has_light_peak:
		roi_mask = None
	elif dominant_peak <= 11:
		roi_mask = cv2.inRange(v_channel, 0, high_v)
	elif dominant_peak >= 20:
		roi_mask = cv2.inRange(v_channel, low_v, 255)
	else:
		roi_mask = None

	if roi_mask is not None and cv2.countNonZero(roi_mask) == 0:
		roi_mask = None

	roi_hist = cv2.calcHist([hsv_roi], [0, 1, 2], roi_mask, [24, 16, 16], [0, 180, 0, 256, 0, 256])
	roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

	return roi_hist


def open_video_and_first_frame(video_path):
	cap = cv2.VideoCapture(video_path)
	ret, frame = cap.read()
	if not ret:
		print(f'Cannot open video: {video_path}')
		cap.release()
		return None, None

	return cap, frame


def get_initial_roi(frame, preset_roi=None):
	if preset_roi is not None:
		x, y, w, h = preset_roi
		if w > 0 and h > 0:
			return x, y, w, h

	x, y, w, h = cv2.selectROI(frame)
	if w == 0 or h == 0:
		return None

	return x, y, w, h


def track_video(video_path, preset_roi=None, window_name='MeanShift_HSV_AdaptiveV'):
	cap, frame = open_video_and_first_frame(video_path)
	if cap is None:
		return

	roi = get_initial_roi(frame, preset_roi)
	if roi is None:
		cap.release()
		cv2.destroyAllWindows()
		return

	x, y, w, h = roi
	track_window = (x, y, w, h)
	hsv_roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
	roi_hist = build_adaptive_roi_histogram(hsv_roi)
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv], [0, 1, 2], roi_hist, [0, 180, 0, 256, 0, 256], 1)
		_, track_window = cv2.meanShift(dst, track_window, term_crit)

		x, y, w, h = track_window
		result = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 255), 2)
		cv2.imshow(window_name, result)

		key = cv2.waitKey(30) & 0xFF
		if key in (27, ord('q')):
			break

	cap.release()
	cv2.destroyAllWindows()


def track_video_template(video_path, preset_roi=None, window_name='Template_Matching'):
	cap, frame = open_video_and_first_frame(video_path)
	if cap is None:
		return

	roi = get_initial_roi(frame, preset_roi)
	if roi is None:
		cap.release()
		cv2.destroyAllWindows()
		return

	x, y, w, h = roi
	template = frame[y:y + h, x:x + w].copy()
	search_padding_x = max(w, 20)
	search_padding_y = max(h, 20)

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		result = frame.copy()
		frame_height, frame_width = frame.shape[:2]
		search_x0 = max(x - search_padding_x, 0)
		search_y0 = max(y - search_padding_y, 0)
		search_x1 = min(x + w + search_padding_x, frame_width)
		search_y1 = min(y + h + search_padding_y, frame_height)

		search_region = frame[search_y0:search_y1, search_x0:search_x1]
		if search_region.shape[0] < h or search_region.shape[1] < w:
			cv2.putText(result, 'Tracking failure', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.imshow(window_name, result)
			continue

		match_map = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
		_, max_val, _, max_loc = cv2.minMaxLoc(match_map)

		if max_val > 0.35:
			x = search_x0 + max_loc[0]
			y = search_y0 + max_loc[1]
			cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)
			cv2.putText(result, f'Match: {max_val:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
		else:
			cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
			cv2.putText(result, 'Tracking failure', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

		cv2.imshow(window_name, result)

		key = cv2.waitKey(30) & 0xFF
		if key in (27, ord('q')):
			break

	cap.release()
	cv2.destroyAllWindows()


def run_demo():
	demo_videos = [VIDEO_1, VIDEO_2]

	for video_path in demo_videos:
		file_name = os.path.basename(video_path)
		preset_roi = DEMO_ROIS.get(file_name)
		track_video(video_path, preset_roi=preset_roi, window_name=f'Demo - {file_name}')


if __name__ == '__main__':
	print('Оберіть режим:')
	print('1 - Demo')
	print('2 - output_1.mp4 with manual ROI')
	print('3 - output_2.mp4 with manual ROI')
	print('4 - output_1.mp4 with Template Matching')
	print('5 - output_2.mp4 with Template Matching')

	mode = int(input('mode: '))

	if mode == 1:
		run_demo()

	if mode == 2:
		track_video(VIDEO_1, window_name='output_1.mp4')

	if mode == 3:
		track_video(VIDEO_2, window_name='output_2.mp4')

	if mode == 4:
		track_video_template(VIDEO_1, window_name='output_1.mp4 - Template Matching')

	if mode == 5:
		track_video_template(VIDEO_2, window_name='output_2.mp4 - Template Matching')

