import base64
import cv2
import requests
import time


class QwenVision:

    def __init__(self):

        self.model = "qwen2.5vl:3b"
        self.url = "http://localhost:11434/api/chat"

        print("Loading Qwen Vision...")
        print("Qwen Vision Ready!")

    def _encode_image(self, frame):

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not success:
            return None

        return base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

    def _is_garbage(self, text):

        if not text:
            return True

        text = text.strip()

        # Reject the @@@@@ response
        if len(text) >= 5 and set(text) <= {"@"}:
            return True

        # Reject extremely short meaningless responses
        if len(text) < 3:
            return True

        return False

    def describe_frame(self, frame):

        image_base64 = self._encode_image(frame)

        if image_base64 is None:
            return "Unable to process camera image."

        prompt = """
You are BlindAssist, an assistive visual system.

Look at this image and describe only what is clearly visible.

Focus on:
people, obstacles, doors, stairs, chairs, tables,
phones, books, notebooks, papers, pens, pencils,
bottles, bags, electronics, signs and pathways.

For important objects, mention left, center, or right.

Prioritize things useful to a visually impaired person.

Do not guess.
Do not speculate.
Do not describe decorative details.

Return ONE short sentence.
Maximum 30 words.
"""

        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],

            "stream": False,

            # Keep the request small and deterministic.
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096,
                "num_predict": 60
            },

            # Important:
            # unload the model after each request.
            # This gives every vision request a clean model state.
            "keep_alive": 0
        }

        for attempt in range(2):

            try:

                print(
                    f"[QWEN] Request attempt {attempt + 1}/2..."
                )

                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=60
                )

                if response.status_code != 200:

                    print(
                        "[QWEN ERROR]:",
                        response.text
                    )

                    time.sleep(1)
                    continue

                data = response.json()

                message = data.get(
                    "message",
                    {}
                )

                result = message.get(
                    "content",
                    ""
                )

                result = str(result).strip()

                print(
                    "[QWEN DONE]:",
                    data.get("done")
                )

                print(
                    "[QWEN RESPONSE]:",
                    result
                )

                # IMPORTANT:
                # Never allow garbage to reach the speaker.
                if self._is_garbage(result):

                    print(
                        "[QWEN] Garbage response detected."
                    )

                    time.sleep(1)
                    continue

                return result

            except Exception as e:

                print(
                    "[QWEN EXCEPTION]:",
                    repr(e)
                )

                time.sleep(1)

        print(
            "[QWEN] No valid response."
        )

        return ""