import cv2

from modules.camera import Camera
from modules.qwen_vision import QwenVision
from modules.qwen_worker import QwenWorker


print("===================================")
print("      BlindAssist Live Vision")
print("===================================")

print("Starting camera...")
camera = Camera()

print("Starting Qwen...")
qwen = QwenVision()

worker = QwenWorker(
    qwen,
    interval=3.0
)

print("Live vision started.")
print("Press Q to quit.")


try:

    while True:

        frame = camera.get_frame()

        if frame is None:
            print("Camera frame unavailable.")
            break

        # Send newest frame to Qwen
        worker.update_frame(frame)

        # Display live camera
        cv2.imshow(
            "BlindAssist Live Vision",
            frame
        )

        # Quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    worker.stop()

    camera.release()

    cv2.destroyAllWindows()

    print("\nBlindAssist stopped.")