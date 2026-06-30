import numpy as np

FLOOR = 5

def calculate_trichion(direction, start, image, detection_window, reference_patch, max_dist):
    step = 1
    pos = start.astype(float).copy()

    MIN_RUN = max(3, detection_window // 2)

    reference_skin = reference_patch.reshape(-1, 3)
    reference_skin_mean = reference_patch.reshape(-1, 3).mean(axis=0)
    skin_spread = np.linalg.norm(reference_skin - reference_skin_mean, axis=1).std()
    skin_spread = max(skin_spread, FLOOR)
    threshold = 3 * skin_spread

    run = 0
    while 0 <= pos[0] < image.shape[1] and 0 <= pos[1] < image.shape[0] and np.linalg.norm(pos - start) < max_dist:
        patch = calculate_detection_window(detection_window, pos, image)
        patch_skin_mean = patch.reshape(-1, 3).mean(axis=0)

        dist = np.linalg.norm(patch_skin_mean - reference_skin_mean)
        if dist > threshold:
            if run == 0:
                run_start = pos.copy()
            run += 1
            if run >= MIN_RUN:
                return run_start
        else: 
            run = 0
        pos = pos + direction * step
    return None

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