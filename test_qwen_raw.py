import cv2

from modules.camera import Camera
from modules.qwen_vision import QwenVision


camera = Camera()

qwen = QwenVision()

print("Capturing frame...")

frame = camera.get_frame()

if frame is None:

    print("FAILED: No camera frame.")
    
else:

    print("Frame captured.")
    print("Sending frame to Qwen...")

    result = qwen.describe_frame(frame)

    print("\n==============================")
    print("FINAL RESULT:")
    print("==============================")
    print(repr(result))


camera.release()
cv2.destroyAllWindows()