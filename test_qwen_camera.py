import cv2

from modules.camera import Camera
from modules.qwen_vision import QwenVision


print("Starting camera...")

camera = Camera()

print("Starting Qwen...")

qwen = QwenVision()

print("Camera + Qwen ready!")

frame = camera.get_frame()

if frame is None:

    print("Could not capture frame.")

else:

    # Show the captured frame
    cv2.imshow("Camera Frame Sent to Qwen", frame)

    print("Sending camera frame to Qwen...")

    description = qwen.describe_frame(frame)

    print("\n==============================")
    print("QWEN DESCRIPTION:")
    print("==============================")
    print(description)

    print("\nPress any key on the camera window to close.")

    cv2.waitKey(0)

camera.release()
cv2.destroyAllWindows()