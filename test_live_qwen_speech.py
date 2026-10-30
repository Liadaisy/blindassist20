import cv2

from modules.camera import Camera
from modules.qwen_vision import QwenVision
from modules.qwen_worker import QwenWorker
from modules.speaker import Speaker


print("===================================")
print("       BlindAssist AI Vision")
print("===================================")

print("Starting camera...")
camera = Camera()

print("Starting Qwen...")
qwen = QwenVision()

print("Starting speaker...")
speaker = Speaker()

worker = QwenWorker(
    qwen,
    interval=4.0
)

print("BlindAssist is ready.")
print("Press Q to quit.")


try:

    while True:

        frame = camera.get_frame()

        if frame is None:

            print("Camera frame unavailable.")
            break

        # Continuously update the newest frame.
        worker.update_frame(frame)

        # Show live camera.
        cv2.imshow(
            "BlindAssist AI Vision",
            frame
        )

        # Get a NEW Qwen message.
        description = worker.get_description()

        if description:

            print("\n==============================")
            print("AI MESSAGE:")
            print(description)
            print("==============================")

            # Speak only valid Qwen output.
            speaker.speak(description)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    worker.stop()

    camera.release()

    cv2.destroyAllWindows()

    print("\nBlindAssist stopped.")