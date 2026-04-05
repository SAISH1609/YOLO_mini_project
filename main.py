"""
YOLOv8 Object Detection (Image / Video / Webcam)

What this script does:
1) Loads a pretrained Ultralytics YOLOv8 model (default: yolov8n.pt)
2) Accepts input as:
   - webcam index ("0", "1", ...)
   - video file path
   - image file path
3) Detects multiple objects in each frame/image
4) Draws bounding boxes + labels
5) Measures and displays detection speed (FPS)

Quick run examples (PowerShell):
- Webcam: python .\main.py --source 0
- Video : python .\main.py --source ".\video.mp4"
- Image : python .\main.py --source ".\image.jpg"
"""

# argparse: reads command-line arguments (--source, --model, --conf)
import argparse
# os: helps with path and file extension checks
import os
# time: used to measure inference speed
import time
# cv2 (OpenCV): used for reading images/video, drawing text, showing windows
import cv2
# YOLO class from Ultralytics library for loading pretrained model and inference
from ultralytics import YOLO

# Set of image file extensions we treat as "single-image input"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: str) -> bool:
    """
    Check whether the given path is an image file based on extension.
    Example:
      "photo.jpg" -> True
      "clip.mp4"  -> False
    """
    ext = os.path.splitext(path.lower())[1]
    return ext in IMAGE_EXTS


def parse_source(src: str):
    """
    Convert source string to the correct type:
    - If user passes digits like "0", convert to int(0) for webcam input.
    - Otherwise keep as string (file path or stream URL).
    """
    return int(src) if src.isdigit() else src


def run_image(model, image_path: str, conf: float):
    """
    Run object detection on a single image.

    Steps:
    1) Read image from disk
    2) Run YOLO inference
    3) Draw bounding boxes/labels
    4) Print object count + timing
    5) Display output image window
    """
    # Read image into a NumPy array
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: cannot read image -> {image_path}")
        return

    # Start timer before model inference
    start_time = time.perf_counter()
    # Run object detection
    results = model(frame, conf=conf, verbose=False)
    # End timer after model inference
    end_time = time.perf_counter()

    # Draw predicted boxes and class labels on image
    annotated = results[0].plot()

    # Count detected objects in this image
    count = len(results[0].boxes) if results[0].boxes is not None else 0
    # Convert inference time to milliseconds
    inference_time = end_time - start_time
    ms = inference_time * 1000

    # Draw image-mode metrics directly on the output image
    cv2.putText(annotated, f"Objects: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"Inference: {ms:.2f} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show annotated image and wait for any key to close
    cv2.imshow("YOLOv8 Detection (Image)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_stream(model, source, conf: float):
    """
    Run object detection continuously on webcam/video stream.

    Steps per frame:
    1) Read frame
    2) Run YOLO inference
    3) Draw boxes/labels
    4) Compute instant FPS and average FPS
    5) Overlay metrics and display frame
    Press 'q' to stop.
    """
    # Open video source (webcam index or video file path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source -> {source}")
        return

    total_inference_time = 0.0  # total inference time across frames
    frames = 0        # number of processed frames

    while True:
        # Read one frame from source
        ok, frame = cap.read()
        if not ok:
            # End of video or camera read failure
            break

        # Measure inference time for this frame
        start_time = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)
        end_time = time.perf_counter()

        inference_time = end_time - start_time
        total_inference_time += inference_time
        frames += 1

        # Draw model predictions on current frame
        annotated = results[0].plot()

        # Instant FPS = speed for current frame
        instant_fps = 1.0 / inference_time if inference_time > 0 else 0.0
        # Average FPS = processed frames / total inference time
        avg_fps = frames / total_inference_time if total_inference_time > 0 else 0.0

        # Count objects detected in current frame
        obj_count = len(results[0].boxes) if results[0].boxes is not None else 0

        # Write metrics on displayed frame
        cv2.putText(annotated, f"Objects: {obj_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"FPS: {instant_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"Avg FPS: {avg_fps:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show annotated live output
        cv2.imshow("YOLOv8 Detection (Video/Webcam)", annotated)

        # Exit when user presses q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanly release resources
    cap.release()
    cv2.destroyAllWindows()

    # No terminal summary: metrics are shown directly on frames.


def main():
    """
    Entry-point function:
    - Parse command-line options
    - Load pretrained model
    - Decide image mode or stream mode
    """
    parser = argparse.ArgumentParser(
        description="Simple YOLOv8 object detection"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Input source: webcam index ("0"), video path, or image path',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO model file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    args = parser.parse_args()

    # Load pretrained YOLO model (downloads model automatically first time)
    model = YOLO(args.model)

    # Convert source string to webcam int if needed
    source = parse_source(args.source)

    # If source is an existing image file, run single-image pipeline
    if isinstance(source, str) and os.path.isfile(source) and is_image(source):
        run_image(model, source, args.conf)
    else:
        # Otherwise run continuous stream pipeline (webcam/video)
        run_stream(model, source, args.conf)


# Python standard entry point
if __name__ == "__main__":
    main()
    
