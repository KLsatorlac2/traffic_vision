from ultralytics import YOLO


class VehicleDetector:
    """
    YOLO vehicle detector.

    Detect:
        car
        motorcycle
        bus
        truck
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path="yolo11n.pt",
        device=0,
        conf=0.25,
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def predict(self, frame, imgsz=640):
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf,
            verbose=False,
            imgsz=imgsz,
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(
            boxes,
            confidences,
            classes,
        ):
            if class_id not in self.VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                    "class_id": class_id,
                    "class_name": self.VEHICLE_CLASSES[class_id],
                }
            )

        return detections