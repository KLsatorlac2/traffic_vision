# ====================================================================================================
# Detection Failure vs Association Failure Analysis
# ====================================================================================================

import cv2
from pathlib import Path
import math

from tracking.tracker import ByteTracker


# ====================================================================================================
# Configuration
# ====================================================================================================

VIDEO_PATH = Path(
    "datasets/videos/traffic.mp4"
)

OUTPUT_DIR = Path(
    "runs/lost_analysis"
)

MODEL_PATH = "yolo11n.pt"

DEVICE = 0

CONF = 0.15

IMG_SIZE = 1280

# IoU threshold for considering a detection spatially close
IOU_THRESHOLD = 0.10

# Center-distance threshold as a ratio of target box diagonal
CENTER_DISTANCE_RATIO = 1.5


# ====================================================================================================
# Lost Gap Cases
# ====================================================================================================

CASES = [
    {
        "track_id": 185,
        "start": 104,
        "end": 139,
    },
    {
        "track_id": 26,
        "start": 17,
        "end": 48,
    },
    {
        "track_id": 101,
        "start": 37,
        "end": 67,
    },
    {
        "track_id": 16,
        "start": 82,
        "end": 102,
    },
    {
        "track_id": 15,
        "start": 79,
        "end": 104,
    },
]


# ====================================================================================================
# IoU
# ====================================================================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1,
    )

    inter_y1 = max(
        ay1,
        by1,
    )

    inter_x2 = min(
        ax2,
        bx2,
    )

    inter_y2 = min(
        ay2,
        by2,
    )

    inter_width = max(
        0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = max(
        0,
        ax2 - ax1,
    ) * max(
        0,
        ay2 - ay1,
    )

    area_b = max(
        0,
        bx2 - bx1,
    ) * max(
        0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


# ====================================================================================================
# Center Distance
# ====================================================================================================

def center_distance(
    box_a,
    box_b,
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    acx = (
        ax1 + ax2
    ) / 2

    acy = (
        ay1 + ay2
    ) / 2

    bcx = (
        bx1 + bx2
    ) / 2

    bcy = (
        by1 + by2
    ) / 2

    return math.sqrt(
        (acx - bcx) ** 2
        + (acy - bcy) ** 2
    )


# ====================================================================================================
# Find nearby YOLO detections
# ====================================================================================================

def find_candidates(
    target_bbox,
    detections,
):

    candidates = []

    if target_bbox is None:
        return candidates

    x1, y1, x2, y2 = target_bbox

    diagonal = math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )

    for detection in detections:

        if detection["class_name"] not in {
            "car",
            "truck",
            "bus",
            "motorcycle",
        }:
            continue

        bbox = detection["bbox"]

        iou = calculate_iou(
            target_bbox,
            bbox,
        )

        distance = center_distance(
            target_bbox,
            bbox,
        )

        normalized_distance = (
            distance / diagonal
            if diagonal > 0
            else float("inf")
        )

        if (
            iou >= IOU_THRESHOLD
            or normalized_distance
            <= CENTER_DISTANCE_RATIO
        ):

            candidates.append(
                {
                    "bbox": bbox,
                    "track_id": detection[
                        "track_id"
                    ],
                    "confidence": detection[
                        "confidence"
                    ],
                    "class_name": detection[
                        "class_name"
                    ],
                    "iou": iou,
                    "distance": distance,
                    "normalized_distance": (
                        normalized_distance
                    ),
                }
            )

    candidates.sort(
        key=lambda x: (
            x["iou"],
            -x["normalized_distance"],
        ),
        reverse=True,
    )

    return candidates


# ====================================================================================================
# Draw
# ====================================================================================================

def draw_frame(
    frame,
    detections,
    target_id,
    target_bbox,
    frame_number,
    status,
):

    # -----------------------------------------------------------------------------------------------
    # Draw every YOLO detection
    # -----------------------------------------------------------------------------------------------

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection["bbox"],
        )

        track_id = detection[
            "track_id"
        ]

        confidence = detection[
            "confidence"
        ]

        class_name = detection[
            "class_name"
        ]

        if track_id == target_id:

            color = (
                0,
                0,
                255,
            )

            thickness = 4

            label = (
                f"TARGET ID {track_id} "
                f"{confidence:.2f}"
            )

        elif track_id is None:

            color = (
                255,
                0,
                0,
            )

            thickness = 2

            label = (
                f"DET "
                f"{class_name} "
                f"{confidence:.2f}"
            )

        else:

            color = (
                0,
                255,
                0,
            )

            thickness = 2

            label = (
                f"ID {track_id} "
                f"{class_name} "
                f"{confidence:.2f}"
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    y1 - 8,
                    20,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # -----------------------------------------------------------------------------------------------
    # Target previous bbox
    # -----------------------------------------------------------------------------------------------

    if target_bbox is not None:

        x1, y1, x2, y2 = map(
            int,
            target_bbox,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

    # -----------------------------------------------------------------------------------------------
    # Frame information
    # -----------------------------------------------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Target ID: {target_id}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return frame


# ====================================================================================================
# Main
# ====================================================================================================

def main():

    print("=" * 110)
    print("DETECTION FAILURE vs ASSOCIATION FAILURE")
    print("=" * 110)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracker = ByteTracker(
        model_path=MODEL_PATH,
        device=DEVICE,
        conf=CONF,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # ================================================================================================
    # Build frame lookup
    # ================================================================================================

    target_frames = {}

    for case in CASES:

        target_id = case["track_id"]

        start = case["start"]

        end = case["end"]

        # Save several frames inside the Lost gap
        middle = (
            start + end
        ) // 2

        frames = {
            start - 1,
            start,
            middle,
            end,
            end + 1,
        }

        for frame_number in frames:

            if (
                frame_number < 1
                or frame_number > total_frames
            ):
                continue

            target_frames.setdefault(
                frame_number,
                [],
            ).append(case)

    # ================================================================================================
    # Runtime information
    # ================================================================================================

    active_cases = {}

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        result = tracker.track(
            frame,
            imgsz=IMG_SIZE,
        )

        tracks = result[
            "tracks"
        ]

        detections = result[
            "detections"
        ]

        # --------------------------------------------------------------------------------------------
        # Only inspect requested frames
        # --------------------------------------------------------------------------------------------

        if frame_number not in target_frames:
            continue

        for case in target_frames[
            frame_number
        ]:

            target_id = case[
                "track_id"
            ]

            start = case[
                "start"
            ]

            end = case[
                "end"
            ]

            # ----------------------------------------------------------------------------------------
            # Find target's last known bbox
            # ----------------------------------------------------------------------------------------

            target_info = (
                tracker.track_history.get(
                    target_id
                )
            )

            target_bbox = None

            if target_info is not None:

                observed_frames = (
                    target_info[
                        "observed_frames"
                    ]
                )

                previous_frames = [
                    f
                    for f in observed_frames
                    if f < frame_number
                ]

                if previous_frames:

                    last_frame = max(
                        previous_frames
                    )

                    # We need the bbox from the corresponding frame.
                    # Use current stored last_bbox only when target is visible.
                    if (
                        target_id
                        in {
                            track["track_id"]
                            for track in tracks
                        }
                    ):

                        for track in tracks:

                            if (
                                track[
                                    "track_id"
                                ]
                                == target_id
                            ):

                                target_bbox = (
                                    track["bbox"]
                                )

                                break

            # ----------------------------------------------------------------------------------------
            # Find nearby YOLO detections
            # ----------------------------------------------------------------------------------------

            candidates = find_candidates(
                target_bbox,
                detections,
            )

            # ----------------------------------------------------------------------------------------
            # Determine status
            # ----------------------------------------------------------------------------------------

            target_visible = any(
                track["track_id"]
                == target_id
                for track in tracks
            )

            if target_visible:

                status = (
                    "TRACKED"
                )

            elif not candidates:

                status = (
                    "NO_NEARBY_DETECTION"
                )

            else:

                candidate = candidates[
                    0
                ]

                candidate_id = (
                    candidate["track_id"]
                )

                if candidate_id is None:

                    status = (
                        "DETECTED_NO_ID"
                    )

                elif (
                    candidate_id
                    != target_id
                ):

                    status = (
                        f"POSSIBLE_ID_SWITCH_TO_{candidate_id}"
                    )

                else:

                    status = (
                        "TRACKED"
                    )

            # ----------------------------------------------------------------------------------------
            # Determine output directory
            # ----------------------------------------------------------------------------------------

            case_dir = (
                OUTPUT_DIR
                / (
                    f"track_{target_id}"
                    f"_gap_{start}_{end}"
                )
            )

            case_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------------------------------------------------
            # Frame type
            # ----------------------------------------------------------------------------------------

            if frame_number == start - 1:

                frame_type = "before"

            elif frame_number == start:

                frame_type = "lost_start"

            elif frame_number == end:

                frame_type = "lost_end"

            elif frame_number == end + 1:

                frame_type = "after"

            else:

                frame_type = "middle"

            output_frame = frame.copy()

            output_frame = draw_frame(
                output_frame,
                detections,
                target_id,
                target_bbox,
                frame_number,
                status,
            )

            output_path = (
                case_dir
                / (
                    f"{frame_type}_"
                    f"{frame_number}.jpg"
                )
            )

            cv2.imwrite(
                str(output_path),
                output_frame,
            )

            # ----------------------------------------------------------------------------------------
            # Console output
            # ----------------------------------------------------------------------------------------

            print(
                f"Frame {frame_number:<4} | "
                f"Target {target_id:<4} | "
                f"{status}"
            )

            for candidate in candidates[:3]:

                print(
                    f"    Candidate: "
                    f"ID={candidate['track_id']} "
                    f"Conf={candidate['confidence']:.3f} "
                    f"IoU={candidate['iou']:.3f} "
                    f"Dist={candidate['normalized_distance']:.2f}"
                )

    cap.release()

    print()
    print("=" * 110)
    print("ANALYSIS FINISHED")
    print("=" * 110)


if __name__ == "__main__":
    main()