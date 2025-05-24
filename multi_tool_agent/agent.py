from google.adk.agents import Agent
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

def extract_metadata(image_url: str) -> dict:
    """
    Extracts metadata from an image given its URL.

    Args:
        image_url (str): The URL of the image.

    Returns:
        dict: A dictionary containing the extracted metadata.
              Returns an error dictionary if something goes wrong.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        image = Image.open(BytesIO(response.content))
        metadata = {}
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded_tag = TAGS.get(tag, tag)
                metadata[decoded_tag] = str(value)
        metadata["format"] = image.format
        metadata["size"] = image.size
        metadata["mode"] = image.mode
        return {"status": "success", "metadata": metadata}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error_message": f"Error downloading image: {e}"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}
    

# Create the agent wrapping the detect_objects tool
root_agent = Agent(
        name="Image_Analysis_Agent",
        model="gemini-2.0-flash",  
        description="Agent that detects objects in images and also finds the metadata of an image.",
        instruction = (
            "You are an Image Analysis Agent. Your task is to analyze an image and provide both object detection results and any available metadata."
            "Instructions:" 
            "- Input Handling:" 
            " - If the user provides a direct image (e.g., uploaded file), analyze the image directly without using external tools for object detection or metadata extraction."
            " - If the user provides a URL link to an image, use available tools (extract_metadata and detect_objects) to retrieve the image and then perform object detection and metadata extraction."
            "- Object Detection: Identify and list all distinct objects present in the image. For each detected object, provide its name (e.g., 'car', 'tree', 'person')."
            "- Metadata Retrieval: Extract any available metadata associated with the image (e.g., file name, size, dimensions, creation date, EXIF data)."
            "- Output Format: Present your findings clearly and concisely using the following format:" " Object Detection Results:"
            " - [Object 1 name]"
            " - [Object 2 name]"
            " or 'No objects detected.'"
            "" 
            " Metadata Results:"
            " - [Key]: [Value]"
            " - [Key]: [Value]"
            " or 'No metadata available.'"
            "- Error Handling: If you encounter errors (e.g., no image provided, invalid URL, image cannot be accessed, or analysis fails), apologize politely and report the error clearly, specifying the issue and suggesting how the user can resolve it (e.g., provide a valid image or URL)."
    ),
    tools=[extract_metadata, detect_objects]
)
