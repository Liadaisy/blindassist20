import threading
import time


class QwenWorker:

    def __init__(self, qwen, interval=4.0):

        self.qwen = qwen
        self.interval = interval

        self.latest_frame = None
        self.latest_description = ""

        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self.thread.start()

        print("Qwen background worker started.")

    def update_frame(self, frame):

        with self.lock:
            self.latest_frame = frame.copy()

    def get_description(self):

        with self.lock:
            description = self.latest_description
            self.latest_description = ""

            return description

    def _worker_loop(self):

        while self.running:

            time.sleep(self.interval)

            if not self.running:
                break

            with self.lock:
                frame = self.latest_frame

            if frame is None:
                continue

            print("\n[QWEN] Analyzing new frame...")

            try:

                description = self.qwen.describe_frame(
                    frame
                )

                if description:

                    with self.lock:
                        self.latest_description = description

            except Exception as e:

                print(
                    "[QWEN WORKER ERROR]:",
                    repr(e)
                )

    def stop(self):

        self.running = False