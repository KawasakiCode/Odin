import cv2
import numpy as np

FLOOR = 5
K = 3

def calculate_trichion(direction, start, image, detection_window, reference_patch, max_dist):
    step = 1
    pos = start.astype(float).copy()

    reason = None
    run_start = None

    MIN_RUN = max(3, detection_window // 2)

    data = []

    alpha = 0.15
    lab_ref = cv2.cvtColor(reference_patch, cv2.COLOR_BGR2LAB)
    ref_lab = lab_ref.reshape(-1, 3)
    ref_lab_mean = ref_lab.mean(axis=0)
    skin_spread = max(np.linalg.norm(ref_lab - ref_lab_mean, axis=1).std(), FLOOR)
    threshold = K * skin_spread

    run = 0
    while 0 <= pos[0] < image.shape[1] and 0 <= pos[1] < image.shape[0] and np.linalg.norm(pos - start) < max_dist:
        patch = calculate_detection_window(detection_window, pos, image)

        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        lab_mean = lab.reshape(-1, 3).mean(axis=0)
        dist = np.linalg.norm(lab_mean - ref_lab_mean)

        if dist <= threshold:
            run = 0
            ref_lab_mean = alpha * lab_mean + (1 - alpha) * ref_lab_mean
        else:
            if run == 0:
                run_start = pos.copy()
            run += 1
            if run >= MIN_RUN:
                return run_start, data, reason
            
        pos = pos + direction * step
        data.append({
            "point": run_start,
            "profile": [pos, dist],
            "start": start,
            "max_dist": max_dist
        })
    if not (0 <= pos[0] < image.shape[1] and 0 <= pos[1] < image.shape[0]):
        reason = "out_of_bounds"
    elif not (np.linalg.norm(pos - start) < max_dist):
        reason = "above_max_dist"
    return None, data, reason

def calculate_detection_window(detection_window, pos, image):
    round_down_to_odd = lambda x : int((x-1) // 2) * 2 + 1

    detection_window = round_down_to_odd(detection_window)
    if detection_window < 7:
        detection_window = 7
    
    half = detection_window // 2
    cx, cy = int(round(pos[0])), int(round(pos[1]))

    x0, x1 = max(0, cx - half), min(image.shape[1], cx + half + 1)
    y0, y1 = max(0, cy - half), min(image.shape[0], cy + half + 1)

    patch = image[y0:y1,x0:x1] # (detection_window, detection_window, 3)

    return patch