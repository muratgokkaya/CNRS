import cv2
import json
import uuid
import threading
import queue
import os
import cProfile
import pstats
from datetime import datetime
import imutils
import pytz
import numpy as np
import torch
import torchvision
from torch.quantization import quantize_dynamic, default_dynamic_qconfig
from ultralytics import YOLO
from flask import Flask, send_from_directory, render_template, make_response
from mjpeg_streamer import MjpegServer, Stream
import logging
from sklearn.cluster import KMeans

# Import from custom modules
from image_processing import enhance_frame, scale_cropped_image
from recognition import Recognizer, Event, calculate_iou, DictObject
from shared_data import ocr_results
#from container_validation import format

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Load configuration from file
with open('config.json', 'r') as config_file:
    CONFIG = json.load(config_file)

# Flatten the config for easier access while maintaining original structure
FLAT_CONFIG = {}
for group in CONFIG:
    FLAT_CONFIG.update(CONFIG[group])

# Ensure directories exist
os.makedirs(FLAT_CONFIG["cropped_images_dir"], exist_ok=True)
os.makedirs(FLAT_CONFIG["still_shots_dir"], exist_ok=True)

# Load existing results from results.json if it exists
RESULTS_FILE = 'results.json'
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        ocr_results[:] = json.load(f)
else:
    ocr_results[:] = []

# Flask routes remain unchanged
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/results.json')
def serve_results():
    response = make_response(send_from_directory(os.path.abspath('.'), 'results.json'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/cropped_images/<path:filename>')
def serve_cropped_image(filename):
    return send_from_directory(FLAT_CONFIG["cropped_images_dir"], filename)

@app.route('/still_shots/<path:filename>')
def serve_still_shot(filename):
    return send_from_directory(FLAT_CONFIG["still_shots_dir"], filename)

# Global variables (unchanged)
frame_queue = queue.Queue(maxsize=5)
last_processed_container = None
last_bounding_box = None
last_recognition_result = None
recognized_container_numbers = []
motion_detected = False
frame_counter = 0
container_processed = False
container_in_frame = False
motion_buffer = []
MOTION_CONFIRMATION_FRAMES = 5
MIN_MOTION_SCORE_FOR_RESET = 50000

# Initialize models with GPU support if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(FLAT_CONFIG["detection_model_path"]).to(device)
modelRecognition = YOLO(FLAT_CONFIG["recognition_model_path"]).to(device)

def quantize_model(model, quantized_layers=None):
    if quantized_layers is None:
        quantized_layers = [torch.nn.Conv2d, torch.nn.Linear]
    qconfig_dict = {layer_type: default_dynamic_qconfig for layer_type in quantized_layers}
    return quantize_dynamic(model, qconfig_dict, dtype=torch.qint8)

# Quantize models
model = quantize_model(model)
modelRecognition = quantize_model(modelRecognition)

# Initialize MJPEG streams (unchanged)
detectionStream = Stream("detection", size=(FLAT_CONFIG["target_frame_width"], FLAT_CONFIG["target_frame_height"]), 
                        quality=FLAT_CONFIG["quality"], fps=FLAT_CONFIG["fps"])
recognitionStream = Stream("recognition", size=(FLAT_CONFIG["target_frame_width"], FLAT_CONFIG["target_frame_height"]), 
                          quality=FLAT_CONFIG["quality"], fps=FLAT_CONFIG["fps"])
server = MjpegServer("*", 8080)
server.add_stream(detectionStream)
server.add_stream(recognitionStream)
server.start()

# Initialize recognizer
recognizer = Recognizer(modelRecognition, recognitionStream)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

def capture_url(url):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FLAT_CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAT_CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, FLAT_CONFIG["fps"])
    return cap

def non_max_suppression(boxes, scores, iou_threshold=0.4):
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    return keep_indices.tolist()

def get_valid_imgsz(width, height, stride=32):
    valid_width = (width + stride - 1) // stride * stride
    valid_height = (height + stride - 1) // stride * stride
    return valid_width, valid_height

def create_parallelogram_mask(frame, points):
    """Create a mask for the parallelogram ROI."""
    mask = np.zeros_like(frame, dtype=np.uint8)
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    return mask

def detect_motion(frame):
    global motion_detected, frame_counter, motion_buffer

    frame_counter += 1
    if frame_counter % FLAT_CONFIG["motion_skip_frames"] != 0:
        return motion_detected

    # Apply parallelogram ROI for motion detection
    points = [
        (FLAT_CONFIG["roi_x1"], FLAT_CONFIG["roi_y1"]),
        (FLAT_CONFIG["roi_x2"], FLAT_CONFIG["roi_y2"]),
        (FLAT_CONFIG["roi_x3"], FLAT_CONFIG["roi_y3"]),
        (FLAT_CONFIG["roi_x4"], FLAT_CONFIG["roi_y4"])
    ]
    mask = create_parallelogram_mask(frame, points)
    roi_frame = cv2.bitwise_and(frame, mask)

    # Crop to bounding box of parallelogram for efficiency
    x_coords, y_coords = zip(*points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Validate cropping bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(FLAT_CONFIG["target_frame_width"], x_max)
    y_max = min(FLAT_CONFIG["target_frame_height"], y_max)

    if x_max <= x_min or y_max <= y_min:
        print(f"Invalid motion ROI bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        return motion_detected

    roi_frame = roi_frame[y_min:y_max, x_min:x_max]
    if roi_frame.size == 0:
        print(f"Empty motion ROI: shape={roi_frame.shape}")
        return motion_detected

    # Resize only if valid
    downsample_width = max(1, roi_frame.shape[1] // FLAT_CONFIG["motion_downsample_factor"])
    downsample_height = max(1, roi_frame.shape[0] // FLAT_CONFIG["motion_downsample_factor"])
    small_roi = cv2.resize(roi_frame, (downsample_width, downsample_height))
    
    fg_mask = bg_subtractor.apply(small_roi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    motion_score = np.sum(fg_mask)
    motion_buffer.append(motion_score > FLAT_CONFIG["motion_threshold"])
    if len(motion_buffer) > MOTION_CONFIRMATION_FRAMES:
        motion_buffer.pop(0)

    motion_detected = (len(motion_buffer) == MOTION_CONFIRMATION_FRAMES and all(motion_buffer))
    
    # Clear buffer if motion is no longer significant
    if not motion_detected:
        motion_buffer.clear()

    # Add a cooldown period after significant motion is detected
    if motion_detected and motion_score > MIN_MOTION_SCORE_FOR_RESET:
        print(f"Significant motion confirmed (score: {motion_score}).")
        motion_buffer.clear()  # Prevent repeated triggers
    return motion_detected

# Additional global variables
container_absent_frames = 0
CONTAINER_ABSENT_THRESHOLD = 5

def save_results_to_file():
    with open(RESULTS_FILE, 'w') as f:
        json.dump(ocr_results, f, indent=4)

def get_ctnr_color(ctnr_img: np.ndarray, num_clusters: int = 3) -> list:
    """
    Get the most dominant color from the image, excluding text and artifacts.
    
    Args:
        ctnr_img (np.ndarray): Input image (cropped container region).
        num_clusters (int): Number of color clusters to use in K-Means.
    
    Returns:
        list: Dominant color as [B, G, R].
    """
    if ctnr_img.size == 0:
        logging.warning("Cropped image is empty. Returning default color (black).")
        return [0, 0, 0]  # Default color (black)

    # Step 1: Resize and blur the image to reduce noise and text impact
    resized_img = cv2.resize(ctnr_img, (100, 100))  # Resize for faster processing
    blurred_img = cv2.GaussianBlur(resized_img, (15, 15), 0)  # Apply Gaussian blur

    # Step 2: Reshape the image to a 2D array of pixels
    pixel_data = blurred_img.reshape((-1, 3))  # Shape: (N, 3), where N = width * height

    # Step 3: Use K-Means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)  # Use n_init to suppress warning
    kmeans.fit(pixel_data)

    # Step 4: Get the cluster centroids and their sizes
    cluster_centers = kmeans.cluster_centers_.astype(int)  # Dominant colors
    cluster_sizes = np.bincount(kmeans.labels_)  # Number of pixels in each cluster

    # Step 5: Exclude very bright or very dark clusters (likely text or shadows)
    brightness_threshold = 50  # Minimum brightness to consider
    valid_clusters = [
        (center, size) for center, size in zip(cluster_centers, cluster_sizes)
        if np.mean(center) > brightness_threshold
    ]

    if not valid_clusters:
        logging.warning("No valid clusters found. Returning default color (black).")
        return [0, 0, 0]  # Default color (black)

    # Step 6: Select the most dominant valid cluster
    dominant_color = max(valid_clusters, key=lambda x: x[1])[0]  # Color with the largest cluster size
    dominant_color = dominant_color.tolist()  # Convert to list [B, G, R]

    logging.info(f"Extracted dominant color: {dominant_color}")  # Debug log
    return dominant_color

def process_frame(frame):
    """Process a frame for container detection and recognition with optional parallelogram ROI."""
    global last_bounding_box, last_recognition_result, motion_detected, last_processed_container
    global container_processed, container_in_frame, container_absent_frames

    try:
        # Preprocess frame
        frame = imutils.rotate(frame, FLAT_CONFIG["rotation_degree"])
        enhanced_frame = enhance_frame(frame)

        if FLAT_CONFIG["use_roi"]:
            # Define parallelogram ROI points
            points = [
                (FLAT_CONFIG["roi_x1"], FLAT_CONFIG["roi_y1"]),
                (FLAT_CONFIG["roi_x2"], FLAT_CONFIG["roi_y2"]),
                (FLAT_CONFIG["roi_x3"], FLAT_CONFIG["roi_y3"]),
                (FLAT_CONFIG["roi_x4"], FLAT_CONFIG["roi_y4"])
            ]

            # Draw parallelogram ROI on the frame
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(enhanced_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            # Create mask and apply it
            mask = create_parallelogram_mask(frame, points)
            roi_frame = cv2.bitwise_and(enhanced_frame, mask)

            # Crop to bounding box of parallelogram
            x_coords, y_coords = zip(*points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Validate cropping bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(FLAT_CONFIG["target_frame_width"], x_max)
            y_max = min(FLAT_CONFIG["target_frame_height"], y_max)

            if x_max <= x_min or y_max <= y_min:
                logging.warning(f"Invalid ROI bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                detectionStream.set_frame(enhanced_frame)
                return

            roi_frame_cropped = roi_frame[y_min:y_max, x_min:x_max]
            if roi_frame_cropped.size == 0:
                logging.warning(f"Empty cropped ROI: shape={roi_frame_cropped.shape}")
                detectionStream.set_frame(enhanced_frame)
                return

            # Adjust image size for model
            valid_imgsz = get_valid_imgsz(x_max - x_min, y_max - y_min)
            resized_roi_frame = cv2.resize(roi_frame_cropped, valid_imgsz)
        else:
            # Use the entire frame if ROI is disabled
            valid_imgsz = get_valid_imgsz(FLAT_CONFIG["target_frame_width"], FLAT_CONFIG["target_frame_height"])
            resized_roi_frame = cv2.resize(enhanced_frame, valid_imgsz)

        # Perform detection
        results = model(resized_roi_frame, imgsz=valid_imgsz, verbose=False, conf=0.4)
        summary = [DictObject.from_dict(item) for item in results[0].summary()]

        if not summary:
            if container_in_frame:
                container_absent_frames += 1
                if container_absent_frames >= CONTAINER_ABSENT_THRESHOLD:
                    logging.info("Container has left the frame. Resetting state.")
                    container_processed = False
                    container_in_frame = False
                    last_processed_container = None
                    container_absent_frames = 0
            else:
                container_absent_frames = 0
            recognizer.transition(Event("not_detected", "", 1, 0, 0, 0, 0, 0))
        else:
            container_in_frame = True
            container_absent_frames = 0

            boxes = [(getattr(recog.box, "x1"), getattr(recog.box, "y1"), getattr(recog.box, "x2"), getattr(recog.box, "y2"))
                     for recog in summary]
            scores = [getattr(recog, "confidence") for recog in summary]
            keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.4)

            for idx in keep_indices:
                recog = summary[idx]
                name = getattr(recog, "name")
                box = recog.box
                x1, y1, x2, y2 = [int(getattr(box, attr)) for attr in ["x1", "y1", "x2", "y2"]]
                confidence = getattr(recog, "confidence")

                if FLAT_CONFIG["use_roi"]:
                    # Scale back to original frame coordinates
                    scale_x = (x_max - x_min) / valid_imgsz[0]
                    scale_y = (y_max - y_min) / valid_imgsz[1]
                    x1, y1, x2, y2 = [int(coord * scale) + offset for coord, scale, offset in
                                      [(x1, scale_x, x_min), (y1, scale_y, y_min),
                                       (x2, scale_x, x_min), (y2, scale_y, y_min)]]
                    x1, y1, x2, y2 = [max(0, min(coord, limit - 1)) for coord, limit in
                                      [(x1, FLAT_CONFIG["target_frame_width"]), (y1, FLAT_CONFIG["target_frame_height"]),
                                       (x2, FLAT_CONFIG["target_frame_width"]), (y2, FLAT_CONFIG["target_frame_height"])]]

                current_bounding_box = (x1, y1, x2, y2)

                if name == "container_number_h" and not container_processed:
                    event = Event("detected", name, confidence, x1, y1, x2, y2, enhanced_frame)
                    recognizer.transition(event)

                    if recognizer.lastRecognition and recognizer.lastRecognition != last_recognition_result:
                        current_container = (recognizer.lastRecognition, current_bounding_box)
                        if last_processed_container:
                            last_num, last_box = last_processed_container
                            iou = calculate_iou(last_box, current_bounding_box)
                            last_center = ((last_box[0] + last_box[2]) // 2, (last_box[1] + last_box[3]) // 2)
                            curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            distance = np.sqrt((last_center[0] - curr_center[0])**2 + (last_center[1] - curr_center[1])**2)

                            # Tighten thresholds to avoid repeated detections
                            if recognizer.lastRecognition == last_num and iou > FLAT_CONFIG["iou_threshold"] and distance < FLAT_CONFIG["min_pixel_shift"]:
                                logging.info(f"Same container detected: {recognizer.lastRecognition}. Skipping OCR.")
                                continue
                            else:
                                container_processed = False  # Reset state for new container

                        logging.info(f"New container recognized: {recognizer.lastRecognition}")
                        last_recognition_result = recognizer.lastRecognition
                        container_processed = True
                        last_processed_container = current_container

                        # Save images and update results
                        current_datetime = datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Istanbul'))
                        timestamp_string = current_datetime.strftime("%Y%m%d%H%M%S")
                        human_readable_timestamp = current_datetime.strftime("%d.%m.%Y - %H:%M:%S")
                        unique_id = uuid.uuid4().hex

                        cropped_image = enhanced_frame[y1:y2, x1:x2]
                        scaled_cropped_image = scale_cropped_image(cropped_image, target_width=600)
                        cropped_filename = f"{unique_id}_{timestamp_string}.jpg"
                        cropped_filepath = os.path.join(FLAT_CONFIG["cropped_images_dir"], cropped_filename)
                        if cv2.imwrite(cropped_filepath, scaled_cropped_image):
                            logging.info(f"Cropped image saved: {cropped_filepath}")

                        still_shot_filepath = os.path.join(FLAT_CONFIG["still_shots_dir"], cropped_filename)
                        if cv2.imwrite(still_shot_filepath, enhanced_frame):
                            logging.info(f"Still shot saved: {still_shot_filepath}")

                        formatted_container_number = ' '.join((recognizer.lastRecognition[:4], recognizer.lastRecognition[4:-1], recognizer.lastRecognition[-1:]))

                        # Get dominant color from the cropped image
                        dominant_color = get_ctnr_color(cropped_image)


                        result = {
                            "unique_id": unique_id,
                            "container_number": formatted_container_number,
                            "cropped_image_path": cropped_filename,
                            "still_shot_path": cropped_filename,
                            "datetime": human_readable_timestamp,
                            "dominant_color": dominant_color
                        }
                        ocr_results.append(result)
                        recognized_container_numbers.append(recognizer.lastRecognition)

                        save_results_to_file()

                    last_bounding_box = current_bounding_box

        # Annotate and stream
        annotated_frame = enhanced_frame.copy()
        if summary:
            for recog in summary:
                if getattr(recog, "name") == "container_number_h":
                    box = recog.box
                    x1, y1, x2, y2 = [int(getattr(box, attr)) for attr in ["x1", "y1", "x2", "y2"]]
                    if FLAT_CONFIG["use_roi"]:
                        x1, y1, x2, y2 = [int(coord * scale) + offset for coord, scale, offset in
                                          [(x1, scale_x, x_min), (y1, scale_y, y_min),
                                           (x2, scale_x, x_min), (y2, scale_y, y_min)]]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"CN Confidence {recog.confidence:.2f}"
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0] + 5, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        if recognizer.lastRecognition:
            # Format the container number
            formatted_container_number = ' '.join((recognizer.lastRecognition[:4], recognizer.lastRecognition[4:-1], recognizer.lastRecognition[-1:]))
            text = formatted_container_number           
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_x = int((annotated_frame.shape[1] - text_size[0]) / 2)
            cv2.rectangle(annotated_frame, (text_x - 5, 30 - text_size[1] - 5), (text_x + text_size[0] + 5, 35), (0, 0, 0), -1)
            cv2.putText(annotated_frame, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw the dominant color square if result is defined
            if ocr_results and "dominant_color" in ocr_results[-1]:
                #logging.info(f"Dominant color: {ocr_results[-1]['dominant_color']}")  # Debug log
                dominant_color = ocr_results[-1]["dominant_color"]
                color_square_size = 30
                color_square_x = text_x + text_size[0] + 10  # Place the square next to the text
                color_square_y = 20 - color_square_size // 2
                cv2.rectangle(annotated_frame,
                              (color_square_x, color_square_y),
                              (color_square_x + color_square_size, color_square_y + color_square_size),
                              tuple(dominant_color), -1)
            else:
                logging.warning("No dominant color found in result.")  # Debug log

        detectionStream.set_frame(annotated_frame)

    except Exception as e:
        logging.error(f"Error processing frame: {e}")

def frame_processor():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                detect_motion(frame)
                process_frame(frame)
            except (BrokenPipeError, ConnectionResetError):
                print("Client disconnected. Stopping frame processing.")
                break
            except Exception as e:
                print(f"Error processing frame: {e}")

def frame_reader(cap):
    global frame_count
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_count += 1
            if frame_count % FLAT_CONFIG["frame_skip"] == 0:
                frame = cv2.resize(frame, (FLAT_CONFIG["target_frame_width"], FLAT_CONFIG["target_frame_height"]))
                if not frame_queue.full():
                    frame_queue.put(frame)
        else:
            break

def main():
    cap = capture_url(FLAT_CONFIG["rtsp"])
    
    reader_thread = threading.Thread(target=frame_reader, args=(cap,))
    reader_thread.start()

    num_processor_threads = 4
    processor_threads = [threading.Thread(target=frame_processor) for _ in range(num_processor_threads)]
    for thread in processor_threads:
        thread.start()

    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000})
    flask_thread.daemon = True
    flask_thread.start()

    reader_thread.join()
    for thread in processor_threads:
        thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    with open('profile_results.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()