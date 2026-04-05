import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from collections import Counter
from PIL import Image
from ultralytics import YOLO

# --- Page Configuration ---
# Using a professional material icon for the browser tab
st.set_page_config(page_title="YOLOv8 Detection", page_icon=":material/visibility:", layout="wide")

# --- SUBTLE STEEL CSS ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* Force main app background to a soft, subtle off-white */
        .stApp, .main {
            background-color: #F4F7F6 !important;
        }
        
        /* Force all general text (and Material Icons) to a highly readable dark slate */
        h1, h2, h3, h4, p, span, div, label {
            color: #2F3640 !important;
        }
        
        /* Sidebar styling - soft muted gray */
        [data-testid="stSidebar"] {
            background-color: #E8ECEB !important;
            border-right: 1px solid #DCDFE0 !important;
        }
        
        /* Force upload box to stay white */
        [data-testid="stFileUploadDropzone"] {
            background-color: #FFFFFF !important;
            border: 2px dashed #CBD5E0 !important;
        }
        
        /* Metric Cards - Pure white with a subtle steel-blue accent */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            padding: 15px !important;
            border-radius: 6px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            border: 1px solid #E2E8F0 !important;
            border-left: 4px solid #5A7D9A !important;
        }
        [data-testid="stMetricLabel"] * {
            color: #718096 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
        }
        [data-testid="stMetricValue"] {
            color: #2C3E50 !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }
        
        /* Buttons - Muted Steel Blue */
        div.stButton > button {
            background-color: #5A7D9A !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
        }
        div.stButton > button:hover {
            background-color: #4A6982 !important;
            color: #FFFFFF !important;
        }
        
        /* Subtle divider lines */
        hr {
            border-color: #DCDFE0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- Load YOLO Model ---
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# --- Sidebar UI ---
st.sidebar.title(":material/settings: Settings")
model_choice = st.sidebar.selectbox("Model:", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
model = load_model(model_choice)
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
# Using Material Icons in the radio buttons
source_type = st.sidebar.radio("Input Source:", (
    ":material/image: Image", 
    ":material/movie: Video", 
    ":material/videocam: Webcam"
))

# --- Main Page UI ---
st.title(":material/troubleshoot: YOLOv8 Object Detection")
st.markdown("---")

def get_class_breakdown(results):
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return {}
    class_ids = results[0].boxes.cls.cpu().tolist()
    class_names = [model.names[int(c)] for c in class_ids]
    return dict(Counter(class_names))

# ==========================================
# 1. IMAGE UPLOAD LOGIC
# ==========================================
if source_type == ":material/image: Image":
    uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            start_time = time.perf_counter()
            results = model(frame_bgr, conf=conf_threshold)
            end_time = time.perf_counter()
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_frame, caption="Detection Result", use_container_width=True)
            
        with col2:
            st.markdown("### :material/monitoring: Metrics")
            obj_count = len(results[0].boxes) if results[0].boxes is not None else 0
            ms = (end_time - start_time) * 1000
            
            st.metric("Objects", obj_count)
            st.metric("Inference", f"{ms:.2f} ms")
            
            st.markdown("### :material/manage_search: Breakdown")
            breakdown = get_class_breakdown(results)
            if breakdown:
                for obj, count in breakdown.items():
                    st.write(f"- **{obj}**: {count}")
            else:
                st.write("No objects detected.")

# ==========================================
# 2. VIDEO UPLOAD LOGIC
# ==========================================
elif source_type == ":material/movie: Video":
    uploaded_video = st.file_uploader("Upload Video...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stframe = st.empty() 
            stop_button = st.button(":material/stop_circle: Stop Video")
            
        with col2:
            st.markdown("### :material/monitoring: Metrics")
            metric_obj = st.empty()
            metric_fps = st.empty()
            metric_avg_fps = st.empty()
            
            st.markdown("### :material/manage_search: Breakdown")
            breakdown_text = st.empty()
            
        total_inference_time = 0.0
        frames = 0
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret: break
                
            start_time = time.perf_counter()
            results = model(frame, conf=conf_threshold, verbose=False)
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            total_inference_time += inference_time
            frames += 1
            
            instant_fps = 1.0 / inference_time if inference_time > 0 else 0.0
            avg_fps = frames / total_inference_time if total_inference_time > 0 else 0.0
            obj_count = len(results[0].boxes) if results[0].boxes is not None else 0
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            metric_obj.metric("Objects", obj_count)
            metric_fps.metric("FPS", f"{instant_fps:.2f}")
            metric_avg_fps.metric("Avg FPS", f"{avg_fps:.2f}")
            
            breakdown = get_class_breakdown(results)
            b_text = ""
            for obj, count in breakdown.items():
                b_text += f"- **{obj}**: {count}\n"
            breakdown_text.markdown(b_text if b_text else "None")
            
        cap.release()

# ==========================================
# 3. WEBCAM LOGIC
# ==========================================
elif source_type == ":material/videocam: Webcam":
    col1, col2 = st.columns([3, 1])
    
    with col1:
        run_webcam = st.checkbox("Start Webcam")
        stframe = st.empty() 
        
    with col2:
        st.markdown("### :material/monitoring: Metrics")
        metric_obj = st.empty()
        metric_fps = st.empty()
        metric_avg_fps = st.empty()
        
        st.markdown("### :material/manage_search: Breakdown")
        breakdown_text = st.empty()
        
    if run_webcam:
        cap = cv2.VideoCapture(0)
        total_inference_time = 0.0
        frames = 0
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret: break
                
            start_time = time.perf_counter()
            results = model(frame, conf=conf_threshold, verbose=False)
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            total_inference_time += inference_time
            frames += 1
            
            instant_fps = 1.0 / inference_time if inference_time > 0 else 0.0
            avg_fps = frames / total_inference_time if total_inference_time > 0 else 0.0
            obj_count = len(results[0].boxes) if results[0].boxes is not None else 0
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            metric_obj.metric("Objects", obj_count)
            metric_fps.metric("FPS", f"{instant_fps:.2f}")
            metric_avg_fps.metric("Avg FPS", f"{avg_fps:.2f}")
            
            breakdown = get_class_breakdown(results)
            b_text = ""
            for obj, count in breakdown.items():
                b_text += f"- **{obj}**: {count}\n"
            breakdown_text.markdown(b_text if b_text else "None")
            
        cap.release()