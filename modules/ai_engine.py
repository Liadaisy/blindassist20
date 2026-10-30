class AIEngine:

    def __init__(self):
        # Higher number = higher priority
        self.priority = {
            "person": 100,
            "stairs": 95,
            "chair": 90,
            "bench": 90,
            "door": 85,
            "bicycle": 80,
            "motorcycle": 80,
            "car": 80,
            "bus": 80,
            "truck": 80,
            "laptop": 50,
            "cell phone": 50,
            "book": 45,
            "bottle": 40,
            "cup": 35,
            "backpack": 35
        }

    def get_position(self, x, width):

        if x < width * 0.33:
            return "left"

        elif x > width * 0.66:
            return "right"

        return "ahead"

    def get_distance(self, area, frame_area):

        ratio = area / frame_area

        if ratio > 0.18:
            return "very close"

        elif ratio > 0.08:
            return "near"

        elif ratio > 0.03:
            return "ahead"

        return "far"

    def generate_message(self, results, detector):

        boxes = results[0].boxes

        if len(boxes) == 0:
            return "No important objects nearby."

        h, w = results[0].orig_shape
        frame_area = h * w

        detections = []

        for box in boxes:

            conf = float(box.conf[0])

            if conf < 0.45:
                continue

            cls = int(box.cls[0])
            name = detector.model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0]

            width = x2 - x1
            height = y2 - y1

            area = width * height

            x_center = (x1 + x2) / 2

            position = self.get_position(x_center, w)
            distance = self.get_distance(area, frame_area)

            score = self.priority.get(name, 1)

            detections.append({
                "name": name,
                "position": position,
                "distance": distance,
                "priority": score
            })

        if len(detections) == 0:
            return "No important objects nearby."

        detections.sort(key=lambda x: x["priority"], reverse=True)

        spoken = set()
        sentences = []

        for obj in detections:

            if obj["name"] in spoken:
                continue

            spoken.add(obj["name"])

            name = obj["name"]
            pos = obj["position"]
            dist = obj["distance"]

            if name == "person":
                sentences.append(
                    f"A person is {dist} {pos}."
                )

            elif name == "chair":
                sentences.append(
                    f"A chair is {dist} on your {pos}."
                )

            elif name == "door":
                sentences.append(
                    f"A doorway is {dist} on your {pos}."
                )

            elif name == "stairs":
                sentences.append(
                    f"Warning. Stairs are {dist} {pos}."
                )

            elif name in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                sentences.append(
                    f"Warning. A {name} is {dist} {pos}."
                )

            else:
                sentences.append(
                    f"A {name} is {dist} on your {pos}."
                )

            # Speak at most 3 important things
            if len(sentences) >= 3:
                break

        return " ".join(sentences)