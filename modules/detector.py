import os
from pathlib import Path
from ultralytics import YOLO


class Detector:

    def __init__(self, model_path=None):
        if model_path is None:
            # Check environment variable first
            env_path = os.environ.get("BLINDASSIST_MODEL_PATH")
            if env_path and os.path.isfile(env_path):
                model_path = env_path
            else:
                # Search candidate locations relative to project root
                project_root = Path(__file__).resolve().parent.parent
                candidates = [
                    project_root / "yolo26s.pt",
                    project_root / "models" / "yolo26s.pt",
                    project_root / "models" / "yolov8s.pt",
                    Path("yolo26s.pt"),
                    Path("models/yolov8s.pt"),
                ]
                for candidate in candidates:
                    if candidate.is_file() and candidate.stat().st_size > 0:
                        model_path = str(candidate)
                        break

            if model_path is None:
                # Fallback to standard yolov8s.pt (Ultralytics auto-downloads if missing)
                print("[WARNING] 'yolo26s.pt' not found locally. Falling back to 'yolov8s.pt'...")
                model_path = "yolov8s.pt"

        print(f"Loading YOLO model from: {model_path}...")
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        print("YOLO model loaded successfully!")

    def detect(self, frame, conf=0.45, imgsz=640):
        results = self.model(
            frame,
            imgsz=imgsz,
            conf=conf,
            verbose=False
        )
        return results
