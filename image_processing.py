import cv2
import numpy as np

def scale_cropped_image(cropped_image, target_width=600, interpolation=cv2.INTER_LINEAR):
    height, width = cropped_image.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    return cv2.resize(cropped_image, (target_width, new_height), interpolation=interpolation)

def apply_clahe(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = clahe.apply(gray_frame)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

def apply_gamma_correction(frame, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def apply_denoising(frame):
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

def apply_color_balance(frame):
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def apply_sharpening(frame):
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

def enhance_frame(frame):
    # Step 1: Dynamic Contrast Adjustment
    ##frame = apply_clahe(frame)  # or apply_gamma_correction(frame, gamma=1.5)

    # Step 2: Noise Reduction
    ##frame = apply_denoising(frame)

    # Step 3: Color Balance
    frame = apply_color_balance(frame)

    # Step 4: Sharpening (optional)
    frame = apply_sharpening(frame)

    return frame
