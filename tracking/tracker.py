# ====================================================================================================
# ByteTrack Vehicle Tracker
# ====================================================================================================

from collections import defaultdict

from ultralytics import YOLO


class ByteTracker:

    # =================================================================================================
    # Initialization
    # =================================================================================================

    def __init__(
        self,
        model_path="yolo11n.pt",
        device=0,
        conf=0.15,
        max_history=30,
    ):
        self.model = YOLO(model_path)

        self.device = device
        self.conf = conf
        self.max_history = max_history

        self.history = defaultdict(list)

        self.track_history = {}

        self.current_frame = 0

    # =================================================================================================
    # Tracking
    # =================================================================================================

    def track(self, frame, imgsz=960):

        self.current_frame += 1

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="configs/bytetrack_custom.yaml",
            conf=self.conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        tracks = []
        detections = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()

            confidences = (
                result.boxes.conf.cpu().numpy()
            )

            classes = (
                result.boxes.cls.cpu().numpy()
            )

            # =========================================================================================
            # Track IDs
            # =========================================================================================

            if result.boxes.id is not None:

                ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                )

            else:

                ids = [None] * len(boxes)

            # =========================================================================================
            # Process all YOLO detections
            # =========================================================================================

            for box, track_id, confidence, cls_id in zip(
                boxes,
                ids,
                confidences,
                classes,
            ):

                confidence = float(confidence)
                cls_id = int(cls_id)

                x1, y1, x2, y2 = map(
                    int,
                    box,
                )

                class_name = self.model.names.get(
                    cls_id,
                    str(cls_id),
                )

                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "track_id": (
                        int(track_id)
                        if track_id is not None
                        else None
                    ),
                }

                # -------------------------------------------------------------------------------------
                # Save every YOLO detection
                # -------------------------------------------------------------------------------------

                detections.append(
                    detection
                )

                # -------------------------------------------------------------------------------------
                # Detection without Track ID
                # -------------------------------------------------------------------------------------

                if track_id is None:
                    continue

                track_id = int(track_id)

                # =====================================================================================
                # Track lifetime
                # =====================================================================================

                if track_id not in self.track_history:

                    self.track_history[track_id] = {
                        "start_frame": self.current_frame,
                        "end_frame": self.current_frame,
                        "length": 1,
                        "confidence_sum": confidence,
                        "observed_frames": [
                            self.current_frame
                        ],
                        "last_bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                    }

                else:

                    info = self.track_history[track_id]

                    info["end_frame"] = (
                        self.current_frame
                    )

                    info["length"] += 1

                    info["confidence_sum"] += (
                        confidence
                    )

                    info["observed_frames"].append(
                        self.current_frame
                    )

                    info["last_bbox"] = [
                        x1,
                        y1,
                        x2,
                        y2,
                    ]

                # =====================================================================================
                # Track center
                # =====================================================================================

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

                self.history[track_id].append(
                    (
                        center_x,
                        center_y,
                    )
                )

                if (
                    len(self.history[track_id])
                    > self.max_history
                ):

                    self.history[track_id] = (
                        self.history[track_id][
                            -self.max_history:
                        ]
                    )

                # -------------------------------------------------------------------------------------
                # Save tracked detection
                # -------------------------------------------------------------------------------------

                tracks.append(
                    detection
                )

        return {
            "tracks": tracks,
            "detections": detections,
        }

    # =================================================================================================
    # Get trajectory history
    # =================================================================================================

    def get_history(self, track_id):

        return self.history.get(
            track_id,
            [],
        )

    # =================================================================================================
    # Basic statistics
    # =================================================================================================

    def get_statistics(self):

        total_tracks = len(
            self.track_history
        )

        if total_tracks == 0:

            return {
                "total_tracks": 0,
                "average_track_length": 0,
                "max_track_length": 0,
                "short_tracks": 0,
            }

        lengths = [
            info["length"]
            for info in self.track_history.values()
        ]

        return {
            "total_tracks": total_tracks,
            "average_track_length": (
                sum(lengths) / len(lengths)
            ),
            "max_track_length": max(lengths),
            "short_tracks": sum(
                length < 10
                for length in lengths
            ),
        }

    # =================================================================================================
    # Track lifetime / Lost statistics
    # =================================================================================================

    def get_track_lifetimes(self):

        results = []

        for track_id, info in (
            self.track_history.items()
        ):

            start_frame = (
                info["start_frame"]
            )

            end_frame = (
                info["end_frame"]
            )

            observed_frames = (
                info["observed_frames"]
            )

            expected_frames = (
                end_frame
                - start_frame
                + 1
            )

            observed_count = len(
                observed_frames
            )

            missing_frames = (
                expected_frames
                - observed_count
            )

            # =========================================================================================
            # Calculate Lost gaps
            # =========================================================================================

            lost_gaps = []

            max_lost_gap = 0

            previous_frame = None

            for frame in observed_frames:

                if previous_frame is not None:

                    gap = (
                        frame
                        - previous_frame
                        - 1
                    )

                    if gap > 0:

                        lost_gaps.append(
                            {
                                "start": (
                                    previous_frame
                                    + 1
                                ),
                                "end": (
                                    frame - 1
                                ),
                                "length": gap,
                            }
                        )

                        max_lost_gap = max(
                            max_lost_gap,
                            gap,
                        )

                previous_frame = frame

            average_confidence = (
                info["confidence_sum"]
                / observed_count
                if observed_count > 0
                else 0
            )

            results.append(
                {
                    "track_id": track_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "length": info["length"],
                    "expected_frames": expected_frames,
                    "observed_frames": observed_count,
                    "missing_frames": missing_frames,
                    "max_lost_gap": max_lost_gap,
                    "lost_gaps": lost_gaps,
                    "average_confidence": average_confidence,
                    "last_bbox": info.get(
                        "last_bbox"
                    ),
                }
            )

        results.sort(
            key=lambda x: x["length"],
            reverse=True,
        )

        return results