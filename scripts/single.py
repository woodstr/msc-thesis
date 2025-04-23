import cv2
import matplotlib.pyplot as plt
import numpy as np


def single_scale_retinex(image, sigma=30):
    image = image.astype(np.float32) + 1.0
    illumination = cv2.GaussianBlur(image, (0, 0), sigma)
    illumination += 1.0
    reflectance = np.log(image) - np.log(illumination)
    reflectance_display = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX)
    reflectance_display = reflectance_display.astype(np.uint8)
    illumination_display = cv2.normalize(illumination, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return reflectance_display, illumination_display


def non_max_suppression_fast(boxes, scores, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[idxs[1:]] - inter)
        idxs = idxs[1:][overlap < overlap_thresh]
    return boxes[keep]


def extract_dominant_dot_template(image, min_area=20, max_area=300, patch_size=(24, 24), offset=5, size_tol=0.5):
    image_clean = cv2.bilateralFilter(image, d=15, sigmaColor=50, sigmaSpace=5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clean = clahe.apply(image_clean)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    tophat = cv2.morphologyEx(image_clean, cv2.MORPH_BLACKHAT, kernel)

    _, binary_top = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    sizes = []
    img_w, img_h = image.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            crop_x_start = x - offset
            crop_x_end = x + w + offset
            crop_y_start = y - offset
            crop_y_end = y + h + offset

            if crop_x_start < 0 or crop_x_end >= img_w or crop_y_start < 0 or crop_y_end >= img_h:
                continue

            patch = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            candidates.append((patch, h, w))
            sizes.append((h, w))

    if not candidates:
        raise ValueError("No valid dot candidates found.")

    # Compute median size
    heights = [s[0] for s in sizes]
    widths = [s[1] for s in sizes]
    median_area = np.median(heights) * np.median(widths)

    # Keep only patches with similar size
    patches_filtered = []
    resized_for_matching = []
    for (patch, h, w) in candidates:
        print(abs(h * w - median_area))
        if abs(h * w - median_area) / median_area < size_tol:
            patches_filtered.append(patch)
            resized_for_matching.append(cv2.resize(patch, patch_size))

    if not patches_filtered:
        raise ValueError("No size-consistent patches found.")

    # Find patch closest to the median template
    stack = np.stack(resized_for_matching, axis=0).astype(np.float32)
    median_template = np.median(stack, axis=0)
    diffs = [np.linalg.norm(p.astype(np.float32) - median_template) for p in resized_for_matching]
    best_idx = np.argmin(diffs)

    return patches_filtered[best_idx], contours

if __name__ == "__main__":
    # === Load image (grayscale) ===
    img = cv2.imread("./data/test2.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (320, 320))

    # === Apply Retinex ===
    reflectance, illumination = single_scale_retinex(img, sigma=64)

    # === Extract dominant template and contours ===
    # REPLACE WITH YOUR ACTUAL TEMPLATE
    dot_template, dot_contours = extract_dominant_dot_template(reflectance)

    # === Template matching ===
    result = cv2.matchTemplate(reflectance, dot_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    locations = zip(*np.where(result >= threshold)[::-1])
    scores = result[result >= threshold].flatten()

    # === Bounding boxes (x, y, w, h) for each match ===
    h, w = dot_template.shape
    boxes = [(int(x), int(y), w, h) for (x, y) in locations]

    # === Apply NMS ===
    nms_boxes = non_max_suppression_fast(boxes, scores, overlap_thresh=0.3)

    # === Draw matching result ===
    output = cv2.cvtColor(reflectance, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in nms_boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # === Draw contours over reflectance ===
    contour_vis = cv2.cvtColor(reflectance, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_vis, dot_contours, -1, (0, 0, 255), 1)

    # === Show results ===
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(illumination, cmap='gray')
    axs[0, 1].set_title("Estimated Illumination")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(reflectance, cmap='gray')
    axs[0, 2].set_title("Reflectance Map (SSR)")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Dot Contours")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(dot_template, cmap='gray')
    axs[1, 1].set_title("Dot template (median of patches)")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Template matching")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()