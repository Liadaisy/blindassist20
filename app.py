"""BlindAssist: AI-Powered Vision Assistance for the Visually Impaired.

A comprehensive Streamlit web application providing real-time object detection,
ByteTrack tracking, threat analysis for approaching obstacles and road hazards,
contextual scene description, and spoken Malayalam voice guidance.
"""

from collections import Counter, defaultdict, deque
import io
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# BlindAssist Core Modules
from modules.detector import Detector
from modules.malayalam import (
    describe_scene_in_malayalam,
    format_threat_warning_in_malayalam,
)
from modules.pipeline import (
    HISTORY_LENGTH,
    draw_visualizations,
    process_frame,
)
from modules.speaker import Speaker
from modules.threat_analyzer import ThreatAnalyzer
from bytetrack_wrapper import MultiClassByteTracker

# Streamlit-WebRTC for browser camera streaming
try:
    from streamlit_webrtc import (
        RTCConfiguration,
        WebRtcMode,
        webrtc_streamer,
    )
    from modules.pipeline import WebRTCBlindAssistProcessor
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    WebRTCBlindAssistProcessor = None

# Optional deep translator fallback for speech playground
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

# ============================================================
# STREAMLIT PAGE CONFIG & CUSTOM THEME
# ============================================================

st.set_page_config(
    page_title="BlindAssist - AI Vision Assistance",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Dark Navy / Blue Theme Styles
CUSTOM_CSS = """
<style>
    /* Dark Navy Accessible Theme */
    .stApp {
        background-color: #0A1128;
        color: #F8FAFC;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #60A5FA !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }
    
    /* Metrics and Cards */
    div[data-testid="stMetric"] {
        background: #131E3D;
        border: 1px solid #1E3A8A;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.95rem !important;
    }

    /* Warning and Hazard Alert Badges */
    .hazard-card-critical {
        background-color: rgba(239, 68, 68, 0.18);
        border-left: 6px solid #EF4444;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
        color: #FECACA;
    }
    .hazard-card-warning {
        background-color: rgba(245, 158, 11, 0.18);
        border-left: 6px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
        color: #FEF3C7;
    }
    .hazard-card-info {
        background-color: rgba(59, 130, 246, 0.18);
        border-left: 6px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
        color: #DBEAFE;
    }
    .malayalam-text {
        font-family: 'Nirmala UI', 'Kartika', 'Gayathri', sans-serif;
        font-size: 1.4rem;
        line-height: 1.6;
        color: #38BDF8;
        font-weight: 600;
    }

    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.6);
        color: #FFFFFF;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# CACHED RESOURCE INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner="Initializing YOLO Computer Vision Model...")
def load_detector():
    """Load and cache the YOLO detection model."""
    try:
        return Detector()
    except Exception as exc:
        st.error(f"Error loading YOLO model: {exc}")
        return None


@st.cache_resource(show_spinner="Initializing Threat & Road Hazard Analyzer...")
def load_threat_analyzer():
    """Load and cache the threat analyzer."""
    return ThreatAnalyzer()


@st.cache_resource(show_spinner="Initializing Malayalam & English Speech Engine...")
def load_speaker():
    """Load and cache the voice synthesis engine."""
    return Speaker()


def get_fresh_tracker(detector_instance):
    """Instantiate a multi-class ByteTracker for a new video or camera session."""
    if detector_instance is None or not hasattr(detector_instance, "model"):
        return None
    names = detector_instance.model.names
    class_names = list(names.values()) if isinstance(names, dict) else names
    return MultiClassByteTracker(
        class_names=class_names,
        track_thresh=0.45,
        track_buffer=30,
        match_thresh=0.8,
        low_thresh=0.1,
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_audio_component(speaker, malayalam_text, key_prefix="audio"):
    """Generate audio bytes and display a browser audio player with download button."""
    if not malayalam_text:
        return

    try:
        audio_bytes = speaker.get_audio_bytes(malayalam_text, lang="ml")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label="📥 Download Malayalam Audio (MP3)",
                data=audio_bytes,
                file_name="blindassist_warning.mp3",
                mime="audio/mpeg",
                key=f"{key_prefix}_download",
            )
        else:
            st.info("Audio generation temporarily unavailable. Malayalam text displayed above.")
    except Exception as exc:
        st.caption(f"Speech synthesis notice: {exc}")


# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================

def main():
    # Initialize Core Engines
    detector = load_detector()
    threat_analyzer = load_threat_analyzer()
    speaker = load_speaker()

    if detector is None:
        st.error(
            "⚠️ The YOLO model weights could not be loaded. Please ensure 'yolo26s.pt' or 'yolov8s.pt' "
            "is available in the project directory."
        )
        st.stop()

    # Sidebar Navigation
    st.sidebar.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/visibility/materialicons/48dp/2x/baseline_visibility_white_48dp.png",
        width=48,
    )
    st.sidebar.title("BlindAssist AI")
    st.sidebar.caption("Vision Assistance for the Visually Impaired")

    nav_options = [
        "🏠 Home & Overview",
        "📹 Live Camera Stream",
        "📸 Snapshot / Mobile Camera",
        "🎬 Video File Analysis",
        "📊 Object Detection Dashboard",
        "⚠️ Threat Analysis & Road Hazards",
        "🎙️ Voice Assistance & Malayalam Speech",
        "ℹ️ Safety & Cloud Deployment",
    ]

    selected_nav = st.sidebar.radio("Navigate Sections", nav_options, index=0)

    # Sidebar Pipeline Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Higher values reduce false detections; lower values detect subtle objects.",
    )
    enable_road_hazards = st.sidebar.checkbox(
        "Detect Kerala Road Hazards (Potholes, Ditches, Puddles)",
        value=True,
        help="Monitors walking corridor for ground surface drops, open ditches, and waterlogged patches.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.85rem; color: #94A3B8;">
            <b>System Status:</b><br>
            • YOLO26s: <span style="color:#22C55E;">Active</span><br>
            • ByteTrack: <span style="color:#22C55E;">Active</span><br>
            • Threat Engine: <span style="color:#22C55E;">Active</span><br>
            • Malayalam TTS: <span style="color:#22C55E;">Online</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # SECTION A: HOME PAGE
    # ========================================================
    if selected_nav == "🏠 Home & Overview":
        st.title("👁️ BlindAssist")
        st.markdown(
            "#### **AI-Powered Vision Assistance for the Visually Impaired**"
        )

        st.markdown(
            """
            BlindAssist transforms camera video streams into real-time spoken guidance and high-contrast
            visual descriptions. Tailored specifically for complex pedestrian environments, the system identifies
            objects, tracks relative velocity to detect rapid approaches, monitors ground hazards (such as open gutters
            and potholes), and speaks clear warnings in **Malayalam** and **English**.
            """
        )

        # Quick Highlights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Detection Architecture", "YOLO26s / v8")
        with col2:
            st.metric("Multi-Object Tracking", "ByteTrack")
        with col3:
            st.metric("Language Support", "Malayalam & EN")
        with col4:
            st.metric("Road Hazard Engine", "Surface & Ditch")

        st.markdown("---")

        # Key Capabilities Grid
        st.subheader("Key Capabilities")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="hazard-card-info">
                    <h4>🚗 Dynamic Obstacle Tracking & Approach Analysis</h4>
                    Tracks pedestrians, cars, buses, bicycles, and staircases across sequential frames.
                    Computes apparent expansion velocity to warn when a vehicle or hazard is rapidly approaching.
                </div>
                <div class="hazard-card-warning">
                    <h4>🕳️ Ground Surface & Road Hazard Detection</h4>
                    Specifically monitors the lower walking corridor for roadside falling hazards (open gutters, ditches, canals),
                    surface depressions (potholes), and specular reflections (slippery puddles).
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="hazard-card-critical">
                    <h4>🗣️ Spoken Malayalam Safety Alerts</h4>
                    Generates immediate Malayalam announcements for urgent hazards (e.g., <i>"ശ്രദ്ധിക്കുക. ഇടതുവശത്ത് ഓടയുണ്ട്. അരികിലേക്ക് പോകരുത്."</i>),
                    keeping warnings actionable without visual clutter.
                </div>
                <div class="hazard-card-info">
                    <h4>📱 Accessible Cloud & Mobile Web Experience</h4>
                    Runs directly in standard desktop and Android mobile web browsers with camera permissions,
                    high-contrast navigation, and zero command-line setup on the phone.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("Get Started")
        start_col1, start_col2, start_col3 = st.columns(3)
        with start_col1:
            if st.button("📹 Open Live Browser Camera", use_container_width=True):
                st.session_state["nav_override"] = "📹 Live Camera Stream"
                st.rerun()
        with start_col2:
            if st.button("📸 Capture Mobile Snapshot", use_container_width=True):
                st.session_state["nav_override"] = "📸 Snapshot / Mobile Camera"
                st.rerun()
        with start_col3:
            if st.button("🎬 Analyze Pre-recorded Video", use_container_width=True):
                st.session_state["nav_override"] = "🎬 Video File Analysis"
                st.rerun()

        # Prototype Safety Notice
        st.markdown(
            """
            > [!CAUTION]
            > **Prototype Safety & Real-World Validation Notice**:
            > BlindAssist is an assistive research and pair-programming prototype. It is **not** a certified medical
            > mobility device, nor does it guarantee collision avoidance or pedestrian safety. Visually impaired users
            > must continue using their primary mobility aids (such as a white cane or guide dog) and exercise full caution.
            """
        )

    # ========================================================
    # SECTION B: LIVE CAMERA DETECTION (WEBRTC STREAMING)
    # ========================================================
    elif selected_nav == "📹 Live Camera Stream":
        st.title("📹 Live Browser Camera Detection")
        st.write(
            "Continuous real-time video stream from your browser camera using WebRTC. "
            "Frames are processed live by YOLO, ByteTrack, and the Threat Analyzer."
        )

        if not WEBRTC_AVAILABLE:
            st.warning(
                "WebRTC streaming component is not available in this environment. "
                "Please use the **Snapshot / Mobile Camera** tab for full browser camera support."
            )
        else:
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            class_names = (
                list(detector.model.names.values())
                if isinstance(detector.model.names, dict)
                else detector.model.names
            )

            # WebRTC Streamer Component
            ctx = webrtc_streamer(
                key="blindassist-live-cam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                video_processor_factory=lambda: WebRTCBlindAssistProcessor(
                    detector=detector,
                    class_names=class_names,
                    conf_thresh=conf_threshold,
                    enable_road_hazards=enable_road_hazards,
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Display real-time telemetry from the live processor
            if ctx.video_processor:
                st.markdown("### Live Scene Telemetry")
                result_placeholder = st.empty()

                with ctx.video_processor.lock:
                    telemetry = ctx.video_processor.latest_result

                if telemetry:
                    t_col1, t_col2, t_col3 = st.columns(3)
                    with t_col1:
                        st.metric("Objects Detected", telemetry.get("detections_count", 0))
                    with t_col2:
                        st.metric("Active Tracks", telemetry.get("tracks_count", 0))
                    with t_col3:
                        st.metric("Hazards / Threats", telemetry.get("threats_count", 0))

                    st.markdown("#### Scene Descriptions")
                    st.info(f"**English:** {telemetry.get('scene_caption', 'Analyzing...')}")
                    st.markdown(
                        f"""
                        <div class="hazard-card-info">
                            <b>മലയാളം വിവരണം:</b><br>
                            <span class="malayalam-text">{telemetry.get('malayalam_caption', 'കാത്തിരിക്കുന്നു...')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if telemetry.get("threat_messages"):
                        st.markdown("#### ⚠️ Active Urgent Warnings")
                        for msg in telemetry["threat_messages"]:
                            st.markdown(
                                f"""
                                <div class="hazard-card-critical">
                                    <span class="malayalam-text">{msg}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            # Audio button for the latest warning
                            get_audio_component(speaker, msg, key_prefix="webrtc")
                else:
                    st.info("Waiting for video frames from camera...")

    # ========================================================
    # SECTION B2: SNAPSHOT / MOBILE CAMERA MODE
    # ========================================================
    elif selected_nav == "📸 Snapshot / Mobile Camera":
        st.title("📸 Mobile & Snapshot Camera Input")
        st.write(
            "Optimized for smartphone mobile browsers (Android / iOS) and firewalled connections. "
            "Click **Take Photo** to capture a frame from your device camera and process it instantly."
        )

        camera_photo = st.camera_input("Capture frame from camera")

        # Also provide a sample image button for quick testing
        sample_col1, sample_col2 = st.columns([1, 4])
        use_sample = False
        with sample_col1:
            if st.button("🧪 Test with Sample Image"):
                use_sample = True

        input_image = None
        if camera_photo is not None:
            input_image = Image.open(camera_photo)
        elif use_sample:
            sample_path = Path("test.jpg")
            if sample_path.exists():
                input_image = Image.open(str(sample_path))
            else:
                st.warning("Sample image 'test.jpg' not found.")

        if input_image is not None:
            # Convert PIL to BGR OpenCV format
            img_np = np.array(input_image)
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif img_np.ndim == 3:
                bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

            tracker = get_fresh_tracker(detector)
            track_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))

            with st.spinner("Analyzing frame with YOLO and Threat Analyzer..."):
                result = process_frame(
                    frame=bgr_frame,
                    detector=detector,
                    tracker=tracker,
                    threat_analyzer=threat_analyzer,
                    track_history=track_history,
                    previous_track_ids=set(),
                    conf_thresh=conf_threshold,
                    enable_road_hazards=enable_road_hazards,
                    draw_visuals=True,
                )

            st.markdown("---")
            view_col1, view_col2 = st.columns([3, 2])

            with view_col1:
                st.subheader("Annotated Detection View")
                st.image(result["annotated_frame_rgb"], use_container_width=True)

            with view_col2:
                st.subheader("Detection Metrics")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Objects Found", len(result["detections"]))
                with m2:
                    st.metric("Hazards Flagged", len(result["threat_warnings"]))

                st.markdown("#### Spoken Malayalam Output")
                combined_msg = result["combined_malayalam"]
                st.markdown(
                    f"""
                    <div class="hazard-card-info">
                        <span class="malayalam-text">{combined_msg}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                get_audio_component(speaker, combined_msg, key_prefix="snapshot")

                st.markdown("#### English Scene Description")
                st.write(result["scene_caption"])

            # Threat and Hazard Summary
            if result["threat_warnings"]:
                st.markdown("### ⚠️ Identified Hazards")
                for warning in result["threat_warnings"]:
                    card_class = "hazard-card-critical" if warning.priority >= 100 else "hazard-card-warning"
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <b>Hazard:</b> {warning.class_name.upper()} | 
                            <b>Zone:</b> {warning.zone.capitalize()} | 
                            <b>Risk Priority:</b> {warning.priority} | 
                            <b>Approaching:</b> {'Yes' if warning.approaching else 'No'}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ========================================================
    # SECTION C: VIDEO FILE ANALYSIS
    # ========================================================
    elif selected_nav == "🎬 Video File Analysis":
        st.title("🎬 Video File Upload & Analysis")
        st.write(
            "Upload recorded video footage (MP4, AVI, MOV) or analyze the included sample road footage. "
            "Frames are processed sequentially without exhausting memory."
        )

        sample_video_path = Path("Data/Person_walking_along_roadside_20261009111619.mp4")
        has_sample = sample_video_path.exists()

        video_choice = st.radio(
            "Select Video Source",
            ["📤 Upload New Video", f"📁 Built-in Road Walking Sample ({sample_video_path.name})"]
            if has_sample
            else ["📤 Upload New Video"],
        )

        video_source_path = None
        if video_choice == "📤 Upload New Video":
            uploaded_video = st.file_uploader(
                "Upload video file",
                type=["mp4", "avi", "mov", "m4v"],
                help="Recommended: 720p or lower for quick processing.",
            )
            if uploaded_video is not None:
                # Save temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                video_source_path = tfile.name
        else:
            video_source_path = str(sample_video_path)

        if video_source_path:
            st.video(video_source_path)

            max_frames_to_process = st.slider(
                "Max frames to process",
                min_value=30,
                max_value=300,
                value=120,
                step=30,
                help="Limits processing duration for fast interactive feedback.",
            )

            if st.button("🚀 Start Video Analysis", type="primary"):
                cap = cv2.VideoCapture(video_source_path)
                if not cap.isOpened():
                    st.error("Unable to open video source.")
                    st.stop()

                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                st.caption(
                    f"Video Properties: {width}x{height} @ {fps:.1f} FPS | Total frames: {total_video_frames}"
                )

                frames_limit = min(max_frames_to_process, total_video_frames if total_video_frames > 0 else max_frames_to_process)

                tracker = get_fresh_tracker(detector)
                track_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
                previous_track_ids = set()

                progress_bar = st.progress(0)
                status_text = st.empty()
                preview_image = st.empty()

                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                m_obj = metrics_col1.empty()
                m_trk = metrics_col2.empty()
                m_haz = metrics_col3.empty()

                live_malayalam_box = st.empty()
                live_english_box = st.empty()

                # Temporary output writer
                out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_path = out_temp.name
                out_temp.close()

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                processed_count = 0
                all_threats_observed = []
                last_spoken_text = None

                try:
                    while processed_count < frames_limit:
                        success, frame = cap.read()
                        if not success or frame is None:
                            break

                        processed_count += 1
                        progress = processed_count / float(frames_limit)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {processed_count} of {frames_limit}...")

                        result = process_frame(
                            frame=frame,
                            detector=detector,
                            tracker=tracker,
                            threat_analyzer=threat_analyzer,
                            track_history=track_history,
                            previous_track_ids=previous_track_ids,
                            conf_thresh=conf_threshold,
                            enable_road_hazards=enable_road_hazards,
                            draw_visuals=True,
                        )

                        if result is None:
                            continue

                        previous_track_ids = result["previous_track_ids"]
                        if result["threat_warnings"]:
                            all_threats_observed.extend(result["threat_warnings"])

                        # Write to output video
                        writer.write(result["annotated_frame_bgr"])

                        # Update UI every 3 frames for smooth rendering
                        if processed_count % 3 == 0 or processed_count == frames_limit:
                            preview_image.image(result["annotated_frame_rgb"], use_container_width=True)
                            m_obj.metric("Current Detections", len(result["detections"]))
                            m_trk.metric("Active Tracks", len(result["tracks"]))
                            m_haz.metric("Hazards Flagged", len(result["threat_warnings"]))

                            live_english_box.info(f"**English:** {result['scene_caption']}")
                            live_malayalam_box.markdown(
                                f"""
                                <div class="hazard-card-info">
                                    <b>മലയാളം വിവരണം:</b><br>
                                    <span class="malayalam-text">{result['combined_malayalam']}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            last_spoken_text = result["combined_malayalam"]

                finally:
                    cap.release()
                    writer.release()

                status_text.success(f"Processing completed! Processed {processed_count} frames.")
                progress_bar.progress(1.0)

                # Audio control for the last generated description
                if last_spoken_text:
                    st.subheader("🔊 Spoken Malayalam Audio")
                    get_audio_component(speaker, last_spoken_text, key_prefix="video_done")

                # Processed video download
                st.subheader("📥 Download Processed Video")
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    with open(out_path, "rb") as vf:
                        st.download_button(
                            label="Download Annotated MP4 Video",
                            data=vf.read(),
                            file_name="blindassist_annotated.mp4",
                            mime="video/mp4",
                        )

    # ========================================================
    # SECTION D: OBJECT DETECTION DASHBOARD
    # ========================================================
    elif selected_nav == "📊 Object Detection Dashboard":
        st.title("📊 Object Detection Dashboard")
        st.write(
            "Inspect detection details, confidence scores, and bounding box coordinates "
            "using the existing 80 COCO-trained YOLO model classes."
        )

        test_img_path = Path("test.jpg")
        uploaded_img = st.file_uploader("Upload an image to inspect", type=["jpg", "jpeg", "png"])

        img_to_inspect = None
        if uploaded_img is not None:
            img_to_inspect = Image.open(uploaded_img)
        elif test_img_path.exists():
            st.info("Displaying default test scene (`test.jpg`). You can also upload custom images above.")
            img_to_inspect = Image.open(str(test_img_path))

        if img_to_inspect is not None:
            np_img = np.array(img_to_inspect)
            bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

            tracker = get_fresh_tracker(detector)
            result = process_frame(
                frame=bgr_img,
                detector=detector,
                tracker=tracker,
                threat_analyzer=threat_analyzer,
                conf_thresh=conf_threshold,
                enable_road_hazards=enable_road_hazards,
                draw_visuals=True,
            )

            # Overview Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Objects", len(result["detections"]))
            with c2:
                st.metric("Distinct Classes", len(result["class_counts"]))
            with c3:
                st.metric("Hazards Flagged", len(result["threat_warnings"]))

            col_view, col_table = st.columns([1, 1])

            with col_view:
                st.subheader("Visual Bounding Boxes")
                st.image(result["annotated_frame_rgb"], use_container_width=True)

            with col_table:
                st.subheader("Class Distribution")
                if result["class_counts"]:
                    df_counts = pd.DataFrame(
                        list(result["class_counts"].items()),
                        columns=["Object Class", "Count"],
                    ).sort_values(by="Count", ascending=False)
                    st.bar_chart(df_counts.set_index("Object Class"))
                else:
                    st.info("No objects detected above the confidence threshold.")

            st.markdown("### Detections Table")
            if result["detections"]:
                det_records = []
                for idx, d in enumerate(result["detections"], 1):
                    x1, y1, x2, y2 = [round(v, 1) for v in d["bbox"]]
                    det_records.append({
                        "#": idx,
                        "Class": d["class_name"],
                        "Confidence": f"{d['confidence']:.2%}",
                        "Bounding Box [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]",
                    })
                st.dataframe(pd.DataFrame(det_records), use_container_width=True)
            else:
                st.info("No detections found.")

    # ========================================================
    # SECTION E: THREAT ANALYSIS & ROAD HAZARDS
    # ========================================================
    elif selected_nav == "⚠️ Threat Analysis & Road Hazards":
        st.title("⚠️ Threat Analysis & Road Hazards")
        st.write(
            "Prioritizes navigational obstacles and Kerala road hazards based on "
            "spatial zones, approach speed, and ground drop severity."
        )

        st.subheader("Threat Hierarchy & Priority Ratings")
        st.markdown(
            """
            The threat analyzer assigns structured priority rankings to obstacles:
            
            | Hazard Class | Base Priority | Threat Type | Danger Description |
            | :--- | :--- | :--- | :--- |
            | **Ditch / Gutter / Canal (*ഓട*)** | **115** | Critical Falling Risk | Open roadside drain edges without barricades |
            | **Stairs (*പടികൾ*)** | **100** | Drop / Tripping Risk | Level differences requiring step awareness |
            | **Pothole (*കുഴി*)** | **98** | Walking Surface Hazard | Deep depressions in walking path |
            | **Approaching Vehicle (*കാർ/ബസ്*)** | **95 + 40 (135)** | Dynamic Collision | Rapidly expanding bounding-box indicating direct approach |
            | **Puddle (*വെള്ളക്കെട്ട്*)** | **70** | Slipping / Wet Hazard | Waterlogged surface patches |
            | **Static Furniture / Door** | **65 - 70** | Navigational Obstacle | Pathways, doors, and corridor obstacles |
            """
        )

        st.markdown("---")
        st.subheader("Interactive Hazard Tester")
        test_hazard_class = st.selectbox(
            "Simulate Obstacle Class",
            ["ditch", "pothole", "puddle", "car", "bus", "stairs", "person"],
        )
        test_zone = st.selectbox("Simulate Spatial Zone", ["center", "left", "right"])
        test_approaching = st.checkbox("Simulate Rapid Approach (Expanding BBox)", value=True)

        warning_malayalam = format_threat_warning_in_malayalam(
            name=test_hazard_class, zone=test_zone, approaching=test_approaching
        )

        st.markdown("#### Generated Spoken Malayalam Warning")
        st.markdown(
            f"""
            <div class="hazard-card-critical">
                <span class="malayalam-text">{warning_malayalam}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        get_audio_component(speaker, warning_malayalam, key_prefix="hazard_test")

    # ========================================================
    # SECTION F: VOICE ASSISTANCE & MALAYALAM SPEECH
    # ========================================================
    elif selected_nav == "🎙️ Voice Assistance & Malayalam Speech":
        st.title("🎙️ Voice Assistance & Malayalam Speech")
        st.write(
            "Test the dual-engine Malayalam voice synthesizer (Edge TTS with gTTS fallback) "
            "and translate custom accessibility descriptions."
        )

        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric(
                "Voice Engine Status",
                "Available" if speaker.has_malayalam_voice() else "Visual Only",
            )
        with status_col2:
            st.metric("Primary Voice", "ml-IN-MidhunNeural (Edge TTS)")

        st.markdown("---")
        st.subheader("🔊 Malayalam Speech Generation Playground")

        sample_phrases = [
            "ശ്രദ്ധിക്കുക. ഇടതുവശത്ത് ഓടയുണ്ട്. അരികിലേക്ക് പോകരുത്.",
            "മുന്നറിയിപ്പ്. മുന്നിൽ റോഡിൽ കുഴിയുണ്ട്. ശ്രദ്ധിച്ചു നടക്കുക.",
            "മുന്നറിയിപ്പ്. കാർ മുന്നിൽ അതിവേഗം അടുത്തുവരുന്നു.",
            "ശ്രദ്ധിക്കുക. മുന്നിൽ വെള്ളക്കെട്ടുണ്ട്.",
            "ദൃശ്യത്തിൽ രണ്ട് ആളുകൾ കാണപ്പെടുന്നു.",
        ]

        preset_choice = st.selectbox("Choose a preset Malayalam alert:", ["Custom Text..."] + sample_phrases)

        if preset_choice != "Custom Text...":
            default_text = preset_choice
        else:
            default_text = "മുന്നറിയിപ്പ്. ശ്രദ്ധിച്ചു നടക്കുക."

        user_text = st.text_area("Malayalam Text to Speak", value=default_text, height=100)

        if st.button("🔊 Synthesize and Speak", type="primary"):
            if not user_text.strip():
                st.warning("Please enter text first.")
            else:
                with st.spinner("Generating speech audio bytes..."):
                    audio_bytes = speaker.get_audio_bytes(user_text.strip(), lang="ml")

                if audio_bytes:
                    st.success("Audio synthesized successfully!")
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    st.download_button(
                        label="Download MP3",
                        data=audio_bytes,
                        file_name="malayalam_alert.mp3",
                        mime="audio/mpeg",
                        key="playground_audio",
                    )
                else:
                    st.error("Audio generation failed. Please verify your internet connection.")

        # Preservation of the English-to-Malayalam translation tool
        if TRANSLATOR_AVAILABLE:
            st.markdown("---")
            st.subheader("🌐 English to Malayalam Translator Tool")
            st.caption("Translate arbitrary English scenes into Malayalam voice output.")

            eng_input = st.text_area(
                "English Scene Description",
                placeholder="Example: A vehicle is approaching rapidly on the right side.",
                height=90,
            )
            if st.button("Translate & Generate Voice"):
                if not eng_input.strip():
                    st.warning("Please enter an English description.")
                else:
                    try:
                        with st.spinner("Translating..."):
                            translated_ml = GoogleTranslator(source="en", target="ml").translate(
                                eng_input.strip()
                            )
                        st.markdown(
                            f"""
                            <div class="hazard-card-info">
                                <b>Malayalam Translation:</b><br>
                                <span class="malayalam-text">{translated_ml}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        get_audio_component(speaker, translated_ml, key_prefix="translator_tool")
                    except Exception as err:
                        st.error(f"Translation service error: {err}")

    # ========================================================
    # SECTION G: SAFETY & CLOUD DEPLOYMENT
    # ========================================================
    elif selected_nav == "ℹ️ Safety & Cloud Deployment":
        st.title("ℹ️ Safety Guidelines & Cloud Deployment")

        st.subheader("🛡️ Safety & Reliability Principles")
        st.markdown(
            """
            1. **Not a Replacement for Primary Mobility Aids**:
               BlindAssist is designed to augment spatial awareness, not replace guide dogs, white canes,
               or professional orientation and mobility training.
            2. **Absence of Detections != Safe Path**:
               Low lighting, lens glare, occlusions, or camera latency can cause objects or ground drops to be missed.
               Users must never assume that silence means a hazard is absent.
            3. **False Positive Mitigation**:
               Warnings de-duplicate identical object-zone pairs to prevent overwhelming the user with continuous speech.
            """
        )

        st.markdown("---")
        st.subheader("☁️ Cloud Deployment Architecture")
        st.markdown(
            """
            BlindAssist is fully structured for one-click deployment on **Streamlit Community Cloud** or **Hugging Face Spaces**:
            
            - **Web Entrypoint**: `app.py`
            - **Configuration**: `.streamlit/config.toml` (dark navy accessible theme)
            - **Python Requirements**: `requirements.txt`
            - **System Dependencies**: `packages.txt` (`libgl1`, `libglib2.0-0`, `ffmpeg`)
            - **Model Weights**: `yolo26s.pt` (20.4MB COCO-trained YOLO model located in repository root)
            - **Relative Pathing**: All paths are resolved dynamically using `pathlib.Path(__file__).resolve().parent`.
            
            #### How to Deploy to Streamlit Community Cloud:
            1. Push the project repository to GitHub:
               ```bash
               git add .
               git commit -m "Convert BlindAssist into Streamlit Cloud Application"
               git push origin main
               ```
            2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
            3. Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
            4. Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` and `packages.txt` and provide a public URL.
            
            #### Mobile Browser Compatibility:
            - **Android Chrome**: Fully supports both **WebRTC Live Stream** and **Snapshot / Mobile Camera** modes.
            - **iOS Safari**: Supports **Snapshot / Mobile Camera** mode natively using the HTML5 camera picker.
            """
        )


if __name__ == "__main__":
    main()
