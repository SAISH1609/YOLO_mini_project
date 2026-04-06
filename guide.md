# YOLOv8 Object Detection Guide

## 1. Open project folder (PowerShell)

```powershell
cd "C:\Users\Saish\Desktop\Sem8\AAI\Project"
```

## 2. Create and activate virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install required packages with the virtual environment activated

```powershell
python -m pip install --upgrade pip
pip install ultralytics opencv-python streamlit numpy pillow
```

## 4. Run the script

### A) General command syntax (all arguments)

```powershell
python .\main.py --source "<SOURCE>" --conf <CONFIDENCE> --model <MODEL_FILE>
```

Example:

```powershell
python .\main.py --source ".\inputs\crowd.jpg" --conf 0.30 --model yolov8s.pt
```

### B) Webcam (default camera)

```powershell
python .\main.py --source 0
```

### C) Second webcam (if connected)

```powershell
python .\main.py --source 1
```

### D) Video file

```powershell
python .\main.py --source ".\inputs\traffic_day.mp4"
```

### E) Image file

```powershell
python .\main.py --source ".\inputs\crowd.jpg"
```

### F) Use another pretrained model

```powershell
python .\main.py --source 0 --model yolov8s.pt
```

### G) Change confidence threshold

```powershell
python .\main.py --source 0 --conf 0.30
```

### H) Use a direct stream URL

```powershell
python .\main.py --source "YOUR_DIRECT_STREAM_URL"
```

Note: Stream URLs work only if `cv2.VideoCapture()` can open them (for example, direct RTSP/direct media links).

### I) Run all files from the `inputs` folder

```powershell
python .\main.py --source ".\inputs\crowd.jpg"
python .\main.py --source ".\inputs\day_street.jpg"
python .\main.py --source ".\inputs\indoor_objects.jpg"
python .\main.py --source ".\inputs\lowlight_motion.mp4"
python .\main.py --source ".\inputs\traffic_day.mp4"
```

### J) Run the frontend (Streamlit app)

```powershell
streamlit run .\app.py
```

After running this command, open the local URL shown in the terminal (usually `http://localhost:8501`).

In the frontend:

- Choose model (`yolov8n.pt`, `yolov8s.pt`, or `yolov8m.pt`) from the sidebar.
- Select input source: Image, Video, or Webcam.
- Upload media for Image/Video mode, or start webcam mode.

## 5. Stop detection

- In webcam/video mode, press **q** in the output window.

## 6. Expected output

- Bounding boxes with class labels on detected objects.
- Object count per frame.
- Instant FPS and average FPS on video/webcam.
- In image mode, inference time is shown on the image.
- In frontend mode, live metrics and class breakdown are shown in the dashboard.
