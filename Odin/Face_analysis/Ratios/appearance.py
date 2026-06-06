"""
Appearance features (skin texture + colour) sampled from the regions defined in
odin_extra.regions.

- Skin texture: variance of the Laplacian inside the forehead/cheek skin
  patches. Higher = more high-frequency detail (pores, wrinkles, blemishes);
  lower = smoother skin.
- Colour: mean colour of the lips and irises as #RRGGBB. For the irises the
  pupil (and any deep shadow) is excluded so its near-black does not wash the
  averaged eye colour lighter — while the threshold stays low enough that a
  genuinely deep-brown iris keeps its true, dark colour.

All functions take a BGR image (OpenCV convention) and pixel-space landmarks.
"""
import cv2
import numpy as np

from odin_extra.regions import extract_regions

# Luminance (0-255, ITU-R BT.601) below which an iris pixel is treated as pupil
# / deep shadow and dropped before averaging. Deliberately low: deep-brown
# irises sit comfortably above ~30, so dark eyes keep their real colour, but
# the pure-black pupil (~0-15) is removed.
IRIS_DARK_LUMA = 30

# BGR -> luma weights (BT.601): luma = 0.299R + 0.587G + 0.114B
_LUMA_BGR = np.array([0.114, 0.587, 0.299])


def _collect_pixels(img_bgr, polygons):
    """Return an Nx3 float array of the BGR pixels inside the given polygon(s)."""
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)
    return img_bgr[mask == 255].astype(np.float64)


def laplacian_texture(img_bgr, polygon):
    """
    Variance of the Laplacian within a single region polygon.

    The Laplacian is computed on the whole grayscale image once, then only the
    in-region pixels are kept for the variance — so the score reflects detail
    inside the skin patch, not the patch's hard mask edge.
    """
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = lap[mask == 255]
    return float(vals.var()) if vals.size else float("nan")


def mean_color(img_bgr, polygons, dark_luma=None):
    """
    Mean colour of one or more regions as an (R, G, B) float tuple.

    polygons: a list of Nx2 int polygons (pooled together — e.g. both irises).
    dark_luma: if set, pixels dimmer than this luma are dropped first (used to
               skip the pupil when averaging iris colour).
    Returns (None, None, None) if no pixels survive.
    """
    px = _collect_pixels(img_bgr, polygons)
    if dark_luma is not None and px.size:
        px = px[px @ _LUMA_BGR >= dark_luma]
    if px.size == 0:
        return (None, None, None)
    b, g, r = px.mean(axis=0)
    return (r, g, b)


def to_hex(rgb):
    """(R, G, B) floats -> '#RRGGBB', or None if any channel is missing."""
    r, g, b = rgb
    if r is None:
        return None
    return "#{:02X}{:02X}{:02X}".format(int(round(r)), int(round(g)), int(round(b)))


def rgb_to_lab(rgb):
    """
    (R, G, B) floats -> standard CIELab (L*, a*, b*).

    OpenCV's 8-bit Lab is scaled (L in 0-255, a/b offset by +128); we rescale to
    the conventional ranges: L* in [0, 100], a*/b* in roughly [-128, 127], so the
    luminance and redness contrasts below are in real CIELab units.
    """
    r, g, b = rgb
    if r is None:
        return (None, None, None)
    px = np.array([[[b, g, r]]], dtype=np.uint8)  # OpenCV wants BGR uint8
    L, a, bb = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
    return (L * 100.0 / 255.0, a - 128.0, bb - 128.0)


def appearance_features(img_bgr, landmarks):
    """
    Compute the colour/texture features for one face, keyed for the CSV.

    img_bgr:   the face image in BGR (OpenCV) order.
    landmarks: Nx3 pixel-space landmark array (same scaling as the image).

    Colours are emitted both as numeric R/G/B channels (what RF/XGB train on)
    and as a convenience #RRGGBB string (human-readable in the CSV; dropped
    before training). Skin colour pools the forehead + both cheeks.
    """
    reg = extract_regions(landmarks)

    # Skin texture: single value, mean Laplacian variance over forehead+cheeks.
    skin_texture = np.nanmean([
        laplacian_texture(img_bgr, reg["forehead"]),
        laplacian_texture(img_bgr, reg["left_cheek"]),
        laplacian_texture(img_bgr, reg["right_cheek"]),
    ])

    lips_rgb = mean_color(img_bgr, [reg["lips"]])
    # Pool both irises into one eye colour, skipping pupil-dark pixels.
    eye_rgb = mean_color(img_bgr, [reg["left_iris"], reg["right_iris"]],
                         dark_luma=IRIS_DARK_LUMA)
    # Skin colour pools the same clean patches used for texture.
    skin_rgb = mean_color(img_bgr, [reg["forehead"], reg["left_cheek"],
                                    reg["right_cheek"]])

    # Facial-contrast features (Russell et al.), computed in CIELab so the
    # luminance (L*) and redness (a*) differences are perceptually meaningful.
    lips_L, lips_a, _ = rgb_to_lab(lips_rgb)
    eye_L,  eye_a,  _ = rgb_to_lab(eye_rgb)
    skin_L, skin_a, _ = rgb_to_lab(skin_rgb)

    def _absdiff(x, y):
        return abs(x - y) if (x is not None and y is not None) else None

    eye_skin_lum = _absdiff(eye_L, skin_L)   # 1. eye-skin luminance contrast
    lip_skin_lum = _absdiff(lips_L, skin_L)  # 2. lip-skin luminance contrast
    lip_skin_red = _absdiff(lips_a, skin_a)  # 3. lip-skin redness contrast
    eye_skin_red = _absdiff(eye_a, skin_a)   # 4. eye-skin redness contrast
    # 5. composite: mean of the two luminance contrasts (Russell's summary).
    facial_contrast_avg = (
        (eye_skin_lum + lip_skin_lum) / 2
        if (eye_skin_lum is not None and lip_skin_lum is not None) else None
    )

    return {
        "skin_texture": float(skin_texture),
        "lips_r": lips_rgb[0], "lips_g": lips_rgb[1], "lips_b": lips_rgb[2],
        "eye_r": eye_rgb[0], "eye_g": eye_rgb[1], "eye_b": eye_rgb[2],
        "skin_r": skin_rgb[0], "skin_g": skin_rgb[1], "skin_b": skin_rgb[2],
        "eye_skin_luminance_contrast": eye_skin_lum,
        "lip_skin_luminance_contrast": lip_skin_lum,
        "lip_skin_redness_contrast": lip_skin_red,
        "eye_skin_redness_contrast": eye_skin_red,
        "facial_contrast_avg": facial_contrast_avg,
    }
