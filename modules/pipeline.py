"""Unified Vision & Assistive Pipeline for BlindAssist.

Integrates YOLO detection, ByteTrack multi-object tracking,
threat analysis (hazard & velocity assessment), offline Malayalam scene
description, and audio warning synthesis.
"""

from collections import Counter, defaultdict, deque
import math
import os
from pathlib import Path
import textwrap
import threading
import time

import cv2
import numpy as np

from modules.detector import Detector
from modules.malayalam import (
    describe_scene_in_malayalam,
    format_threat_warning_in_malayalam,
)
from modules.speaker import Speaker
from modules.threat_analyzer import ThreatAnalyzer
from bytetrack_wrapper import MultiClassByteTracker

try:
    from streamlit_webrtc import VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object

# Constants matching main.py
HISTORY_LENGTH = 20
MOTION_MIN_PIXELS = 12


def yolo_to_tracker_detections(results, detector_instance, min_conf=0.35):
    """Convert Ultralytics YOLO results into the tracker's expected detection format."""
    detections = []
    if results is None or len(results) == 0:
        return detections

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    names = detector_instance.model.names
    for i in range(len(boxes)):
        try:
            confidence = float(boxes.conf[i].item())
            if confidence < min_conf:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            class_id = int(boxes.cls[i].item())
            if isinstance(names, dict):
                class_name = names.get(class_id, str(class_id))
            else:
                class_name = names[class_id]

            detections.append({
                "class_name": str(class_name),
                "class_id": class_id,
                "confidence": confidence,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        except (IndexError, TypeError, ValueError, AttributeError) as exc:
            print(f"[PIPELINE WARNING] Skipping detection {i}: {exc}")
    return detections


def pluralize_object(name, count):
    """Pluralize common object labels."""
    if count == 1:
        return name
    irregular = {"person": "people", "mouse": "mice"}
    if name in irregular:
        return irregular[name]
    if name.endswith(("bus", "class")):
        return name + "es"
    if name.endswith("y") and len(name) > 1 and name[-2].lower() not in "aeiou":
        return name[:-1] + "ies"
    if name.endswith("s"):
        return name
    return name + "s"


def get_position(center_x, frame_width):
    ratio = center_x / max(float(frame_width), 1.0)
    if ratio < 0.34:
        return "left"
    if ratio > 0.66:
        return "right"
    return "center"


def collect_scene_details(tracks, track_history, frame_width, new_track_ids=None):
    unique_tracks = {}
    for track in tracks:
        if not isinstance(track, dict) or "bbox" not in track or "class_name" not in track:
            continue
        track_id = track.get("track_id")
        unique_tracks[track_id if track_id is not None else id(track)] = track

    if not unique_tracks:
        return None

    counts = Counter(str(t.get("class_name", "object")) for t in unique_tracks.values())
    positions = []
    motions = []
    for track_id, track in unique_tracks.items():
        try:
            x1, _, x2, _ = map(float, track["bbox"])
        except (TypeError, ValueError):
            continue

        class_name = str(track.get("class_name", "object"))
        location = get_position((x1 + x2) / 2, frame_width)
        position = (class_name, location)
        if position not in positions:
            positions.append(position)

        real_track_id = track.get("track_id")
        if real_track_id is not None and real_track_id in track_history:
            points = track_history[real_track_id]
            if len(points) >= 5:
                dx = points[-1][0] - points[-5][0]
                if abs(dx) >= max(MOTION_MIN_PIXELS, frame_width * 0.015):
                    direction = "right" if dx > 0 else "left"
                    motion = (class_name, direction)
                    if motion not in motions:
                        motions.append(motion)

        if len(positions) >= 3:
            break

    newly_tracked = []
    if new_track_ids:
        for track in unique_tracks.values():
            if track.get("track_id") in new_track_ids:
                name = str(track.get("class_name", "object"))
                if name not in newly_tracked:
                    newly_tracked.append(name)

    return counts, positions, motions, newly_tracked


def describe_scene_with_context(tracks, track_history, frame_width, new_track_ids=None):
    """Describe tracked object counts, approximate image positions, and horizontal motion."""
    if not tracks:
        return "No objects are currently detected in the frame."

    details = collect_scene_details(
        tracks, track_history, frame_width, new_track_ids
    )
    if details is None:
        return "No tracked objects are currently available."

    counts, positions, motions, newly_tracked = details
    object_summary = [
        f"{count} {pluralize_object(name, count)}"
        for name, count in counts.most_common()
    ]
    object_sentence = (
        f"{object_summary[0]} is visible" if len(object_summary) == 1
        else ", ".join(object_summary[:-1]) + " and " + object_summary[-1] + " are visible"
    )

    caption = f"The scene contains {object_sentence}."
    if positions:
        position_phrases = [
            f"{name} on the {position}" for name, position in positions
        ]
        caption += " Positions: " + "; ".join(position_phrases) + "."
    if motions:
        motion_phrases = [
            f"{name} is moving across the image toward the {direction}"
            for name, direction in motions[:2]
        ]
        caption += " Movement: " + "; ".join(motion_phrases) + "."
    if newly_tracked:
        caption += " Newly tracked: " + ", ".join(newly_tracked) + "."
    return caption


def describe_scene_in_malayalam_with_context(
    tracks, track_history, frame_width, new_track_ids=None
):
    """Build an offline Malayalam caption from the tracked scene details."""
    if not tracks:
        return describe_scene_in_malayalam([], [], [], [])

    details = collect_scene_details(
        tracks, track_history, frame_width, new_track_ids
    )
    if details is None:
        return "ട്രാക്ക് ചെയ്ത വസ്തുക്കൾ ലഭ്യമല്ല."

    counts, positions, motions, newly_tracked = details
    return describe_scene_in_malayalam(
        counts.most_common(), positions, motions[:2], newly_tracked
    )


def draw_caption_banner(frame, caption_text):
    """Draw a readable multi-line caption overlay at the bottom of the video frame."""
    if not caption_text:
        return frame

    h, w = frame.shape[:2]
    lines = textwrap.wrap(caption_text, width=max(40, int(w / 14))) or [caption_text]
    line_height = 24
    panel_height = 14 + line_height * len(lines)
    y_top = max(0, h - panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_top), (w, h), (10, 15, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for index, line in enumerate(lines):
        y = min(h - 6, y_top + 22 + index * line_height)
        cv2.putText(
            frame, line, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def draw_visualizations(
    frame,
    detections,
    tracks,
    track_history,
    threat_warnings,
    scene_caption=None,
    show_caption=True,
    show_hud=True,
):
    """Overlay bounding boxes, tracks, road hazards, and status HUD onto the frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Draw tracks and trajectories
    for track in tracks:
        if not isinstance(track, dict) or "bbox" not in track:
            continue
        try:
            x1, y1, x2, y2 = map(int, track["bbox"])
        except (TypeError, ValueError):
            continue

        class_name = str(track.get("class_name", "object"))
        track_id = track.get("track_id")
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw bounding box and label
        box_color = (255, 120, 0) # Blue in BGR
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
        label = f"{class_name} #{track_id}" if track_id is not None else class_name
        cv2.putText(
            display, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA
        )
        cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

        # Draw trajectory history line
        if track_id is not None and track_id in track_history:
            points = track_history[track_id]
            for i in range(1, len(points)):
                cv2.line(display, points[i - 1], points[i], (255, 120, 0), 2)

    # Draw Road Hazard & Obstacle Warnings
    for warning in threat_warnings or []:
        if getattr(warning, "bbox", None) is not None:
            try:
                bx1, by1, bx2, by2 = map(int, warning.bbox)
                is_critical = warning.class_name in ("ditch", "gutter", "canal") or warning.priority >= 110
                hazard_color = (0, 0, 255) if is_critical else (0, 165, 255) # Red or Orange
                cv2.rectangle(display, (bx1, by1), (bx2, by2), hazard_color, 3)

                hazard_label = f"HAZARD: {warning.class_name.upper()}"
                if warning.approaching:
                    hazard_label += " [APPROACHING!]"

                cv2.putText(
                    display, hazard_label,
                    (bx1, max(25, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hazard_color, 2, cv2.LINE_AA
                )
            except Exception:
                pass

    # HUD Status overlay
    if show_hud:
        hud_bg = display[:105, :260].copy()
        cv2.rectangle(display, (0, 0), (260, 105), (15, 23, 42), -1)
        cv2.addWeighted(hud_bg, 0.35, display[:105, :260], 0.65, 0, display[:105, :260])

        cv2.putText(display, "BlindAssist AI Vision", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2, cv2.LINE_AA)
        cv2.putText(display, f"Detections: {len(detections)}", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(display, f"Active Tracks: {len(tracks)}", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        threat_count = len(threat_warnings or [])
        threat_color = (0, 80, 255) if threat_count > 0 else (0, 230, 118)
        cv2.putText(display, f"Hazards / Threats: {threat_count}", (12, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, threat_color, 2, cv2.LINE_AA)

    if show_caption and scene_caption:
        display = draw_caption_banner(display, scene_caption)

    return display


def process_frame(
    frame,
    detector,
    tracker,
    threat_analyzer,
    track_history=None,
    previous_track_ids=None,
    conf_thresh=0.45,
    enable_road_hazards=True,
    draw_visuals=True,
):
    """Complete BlindAssist processing on a single image frame (BGR format).

    Returns a comprehensive result dictionary.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    if track_history is None:
        track_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
    if previous_track_ids is None:
        previous_track_ids = set()

    h, w = frame.shape[:2]

    # 1. YOLO Inference
    results = detector.detect(frame, conf=conf_thresh)
    detections = yolo_to_tracker_detections(results, detector, min_conf=conf_thresh)

    # 2. ByteTrack Update
    tracks = tracker.update(detections) if tracker else []
    if tracks is None:
        tracks = []

    # 3. Update Tracking History & Detect Newly Tracked Objects
    current_track_ids = set()
    for track in tracks:
        if not isinstance(track, dict) or "bbox" not in track:
            continue
        try:
            x1, y1, x2, y2 = map(int, track["bbox"])
        except (TypeError, ValueError):
            continue

        track_id = track.get("track_id")
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id is not None:
            current_track_ids.add(track_id)
            track_history[track_id].append((cx, cy))

    new_track_ids = current_track_ids - previous_track_ids
    updated_previous_track_ids = current_track_ids

    # 4. Contextual Scene Description (English & Malayalam)
    scene_caption = describe_scene_with_context(
        tracks, track_history, w, new_track_ids=new_track_ids
    )
    malayalam_caption = describe_scene_in_malayalam_with_context(
        tracks, track_history, w, new_track_ids=new_track_ids
    )

    # 5. Threat & Hazard Analysis
    threat_warnings = threat_analyzer.analyze(
        tracks, w, h, frame=frame if enable_road_hazards else None
    )

    threat_messages = [
        format_threat_warning_in_malayalam(
            warning.class_name, warning.zone, warning.approaching
        )
        for warning in threat_warnings
    ]

    combined_malayalam = malayalam_caption
    if threat_messages:
        combined_malayalam += " " + " ".join(threat_messages)

    # 6. Detection Statistics by Class
    class_counts = Counter(d["class_name"] for d in detections)

    # 7. Visualization Overlay
    if draw_visuals:
        annotated_bgr = draw_visualizations(
            frame,
            detections=detections,
            tracks=tracks,
            track_history=track_history,
            threat_warnings=threat_warnings,
            scene_caption=scene_caption,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    else:
        annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    highest_threat = threat_warnings[0] if threat_warnings else None

    return {
        "annotated_frame_rgb": annotated_rgb,
        "annotated_frame_bgr": annotated_bgr if draw_visuals else frame,
        "detections": detections,
        "tracks": tracks,
        "threat_warnings": threat_warnings,
        "highest_threat": highest_threat,
        "threat_messages": threat_messages,
        "scene_caption": scene_caption,
        "malayalam_caption": malayalam_caption,
        "combined_malayalam": combined_malayalam,
        "class_counts": dict(class_counts),
        "previous_track_ids": updated_previous_track_ids,
        "track_history": track_history,
    }


if WEBRTC_AVAILABLE:
    class WebRTCBlindAssistProcessor(VideoProcessorBase):
        """Thread-safe WebRTC Video Processor for real-time browser camera streaming."""

        def __init__(self, detector, class_names, conf_thresh=0.45, enable_road_hazards=True):
            self.detector = detector
            self.tracker = MultiClassByteTracker(
                class_names=class_names,
                track_thresh=0.45,
                track_buffer=25,
                match_thresh=0.8,
                low_thresh=0.1,
            )
            self.threat_analyzer = ThreatAnalyzer()
            self.track_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
            self.previous_track_ids = set()
            self.conf_thresh = conf_thresh
            self.enable_road_hazards = enable_road_hazards

            self.lock = threading.Lock()
            self.latest_result = None
            self.frame_count = 0
            self.last_process_time = time.time()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            result = process_frame(
                frame=img,
                detector=self.detector,
                tracker=self.tracker,
                threat_analyzer=self.threat_analyzer,
                track_history=self.track_history,
                previous_track_ids=self.previous_track_ids,
                conf_thresh=self.conf_thresh,
                enable_road_hazards=self.enable_road_hazards,
                draw_visuals=True,
            )

            if result is not None:
                self.previous_track_ids = result["previous_track_ids"]
                with self.lock:
                    self.latest_result = {
                        "detections_count": len(result["detections"]),
                        "tracks_count": len(result["tracks"]),
                        "threats_count": len(result["threat_warnings"]),
                        "scene_caption": result["scene_caption"],
                        "malayalam_caption": result["malayalam_caption"],
                        "combined_malayalam": result["combined_malayalam"],
                        "threat_messages": result["threat_messages"],
                        "class_counts": result["class_counts"],
                        "highest_threat": (
                            {
                                "class_name": result["highest_threat"].class_name,
                                "zone": result["highest_threat"].zone,
                                "priority": result["highest_threat"].priority,
                                "approaching": result["highest_threat"].approaching,
                            }
                            if result["highest_threat"]
                            else None
                        ),
                    }
                return av.VideoFrame.from_ndarray(result["annotated_frame_bgr"], format="bgr24")

            return frame

