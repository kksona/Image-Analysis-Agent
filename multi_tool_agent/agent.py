import cv2
import numpy as np
from ultralytics import YOLO
import requests
from PIL import Image
from PIL.ExifTags import TAGS
import requests
from io import BytesIO
from google.adk.agents import Agent

# Load the model once globally to avoid reloading on every call
model = YOLO("yolov8n.pt")

def detect_objects(image_url: str) -> dict:
    """
    Detect objects in an image from a URL using YOLOv8.

    Args:
        image_url (str): URL of the image to analyze.

    Returns:
        dict: status and list of detected objects with label, confidence, bbox.
    """
    response = requests.get(image_url)
    if response.status_code != 200:
        return {
            "status": "error",
            "error_message": "Failed to download image from URL"
        }

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

    results = model(image)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

    return {
        "status": "success",
        "objects": detected_objects
    }

def get_metadata(image_url: str) -> dict:
    """
    Detect objects in an image from a URL using YOLOv8.

    Args:
        image_url (str): URL of the image to analyze.

    Returns:
        dict: status and list of detected objects with label, confidence, bbox.
    """
    response = requests.get(image_url)
    if response.status_code != 200:
        return {
            "status": "error",
            "error_message": "Failed to download image from URL"
        }

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

    results = model(image)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

    return {
        "status": "success",
        "objects": detected_objects
    }


# Create the agent wrapping the detect_objects tool
root_agent = Agent(
    name="object_detection_agent",
    model="gemini-2.0-flash",  
    description="Agent that detects objects in images using YOLOv8.",
    instruction=(
        "You are an assistant that detects objects in images given a URL. "
        "Return the detected objects with their label, confidence score, and bounding box coordinates."
    ),
    tools=[detect_objects]
)
