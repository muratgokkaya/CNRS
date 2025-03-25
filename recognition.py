from datetime import datetime, timedelta
import json
# Import from other modules
from container_validation import is_valid_iso_container

class DictObject:
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    @classmethod
    def from_dict(cls, d):
        return json.loads(json.dumps(d), object_hook=DictObject)

class Recognizer:
    def __init__(self, modelRecognition, recognitionStream):
        self.reset()
        self.lastRecognition = ""
        self.lastFailedRecognition = ""
        self.modelRecognition = modelRecognition
        self.recognitionStream = recognitionStream
        self.valid_container_detected = False  # Flag to track valid container detection
        self.last_recog_time = datetime.utcnow() - timedelta(days=1)  # Initialize to a day ago

    def reset(self):
        self.detectedCount = 0
        self.notDetectedCount = 0
        self.recognitions = []

    def recognize(self):
        objectToRecognize = max(self.recognitions, key=lambda obj: obj.confidence)
        x1, y1, x2, y2 = int(objectToRecognize.x1), int(objectToRecognize.y1), int(objectToRecognize.x2), int(objectToRecognize.y2)
        croppedRoi = objectToRecognize.frame[y1:y2, x1:x2]

        bestCandidate = process_recognition(croppedRoi, self.modelRecognition, self.recognitionStream, objectToRecognize.recogClass)
        validation_result = is_valid_iso_container(bestCandidate)  # Use the new validation function
        if validation_result:
            print(f"Recognition successful for container number: {bestCandidate}")
            self.lastRecognition = bestCandidate
            self.valid_container_detected = True  # Set flag to True
        else:
            print(f"Recognition failed for container number: {bestCandidate}. Reason: Invalid check digit.")
            self.lastFailedRecognition = bestCandidate
        return

    def printLastRecogData(self):
        if self.lastFailedRecognition or self.lastRecognition:
            print("OK:", self.lastRecognition, "NOK:", self.lastFailedRecognition)

    def transition(self, event):
        if event.name == "detected":
            self.detectedCount += 1
            self.recognitions.append(event)
        if event.name == "not_detected":
            self.notDetectedCount += 1

        if self.detectedCount > 2:
            self.recognize()
            self.reset()
        if self.notDetectedCount > 2:
            self.reset()
            self.lastRecognition = ""
            self.lastFailedRecognition = ""

    def check_recog_time(self):
        current_time = datetime.utcnow()
        time_diff = current_time - self.last_recog_time
        if time_diff.total_seconds() < 3:  # Check every 3 seconds
            return True
        self.last_recog_time = current_time
        return False

class Event:
    def __init__(self, name, recogClass, confidence, x1, y1, x2, y2, frame):
        self.name = name
        self.confidence = confidence
        self.recogClass = recogClass
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame = frame

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as (x1, y1, x2, y2).
    """
    # Determine the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Compute the area of the intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou

def detect_overlapping_boxes(recog, iou_threshold=0.5):
    overlapping_pairs = {}
    n = len(recog)
    for i in range(n):
        overlapping_pairs[i] = [recog[i]]
    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(
                (getattr(recog[i].box, "x1"), getattr(recog[i].box, "y1"), getattr(recog[i].box, "x2"), getattr(recog[i].box, "y2")),
                (getattr(recog[j].box, "x1"), getattr(recog[j].box, "y1"), getattr(recog[j].box, "x2"), getattr(recog[j].box, "y2"))
            )
            if iou > iou_threshold:
                overlapping_pairs[i].append(recog[j])
    return overlapping_pairs

def process_recognition(croppedRoi, modelRecognition, outputVideoStream, direction='container_number_v'):
    results = modelRecognition.predict(croppedRoi, imgsz=640, line_width=1, show_conf=False, show_labels=True, verbose=False)
    names = modelRecognition.names
    annotated_frame = results[0].plot(line_width=1)
    outputVideoStream.set_frame(annotated_frame)

    resultSummarized = results[0].summary()
    # Convert summary to DictObject instances
    resultSummarized = [DictObject.from_dict(item) for item in resultSummarized]

    if direction == 'container_number_h':
        sorted_detections = sorted(resultSummarized, key=lambda d: getattr(d.box, "x1"), reverse=False)
    elif direction == 'container_number_v':
        sorted_detections = sorted(resultSummarized, key=lambda d: getattr(d.box, "y1"), reverse=False)
    else:
        return ""

    overlapping = detect_overlapping_boxes(sorted_detections)
    ret = []
    for key, value in overlapping.items():
        if len(value) == 1:
            ret.append(names[getattr(value[0], "class")])
    return "".join(ret).upper()
