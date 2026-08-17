import cv2
from pathlib import Path


from tracking.tracker import ByteTracker


# ====================================================================================================
# Configuration
# ====================================================================================================

VIDEO_PATH = Path("datasets/videos/traffic.mp4")

MODEL_PATH = "yolo11n.pt"
DEVICE = 0

CONF = 0.15
IMGSZ_LIST = [1280]

# -----------------------------------------------------------------------------------------------
# ID Switch Detection Parameters
# -----------------------------------------------------------------------------------------------

# Minimum IoU between old and new track
IOU_THRESHOLD = 0.30

# Maximum center distance relative to image diagonal
CENTER_DIST_THRESHOLD = 0.05

# New ID must continue for at least this many frames
MIN_NEW_ID_LENGTH = 3


# ====================================================================================================
# Utility Functions
# ====================================================================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1,
    )

    intersection_height = max(
        0,
        y2 - y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


def calculate_center_distance(
    box1,
    box2,
    frame_width,
    frame_height,
):

    center1_x = (
        box1[0] + box1[2]
    ) / 2

    center1_y = (
        box1[1] + box1[3]
    ) / 2

    center2_x = (
        box2[0] + box2[2]
    ) / 2

    center2_y = (
        box2[1] + box2[3]
    ) / 2

    dx = center1_x - center2_x
    dy = center1_y - center2_y

    distance = (
        dx ** 2
        + dy ** 2
    ) ** 0.5

    diagonal = (
        frame_width ** 2
        + frame_height ** 2
    ) ** 0.5

    if diagonal <= 0:
        return 1.0

    return distance / diagonal


# ====================================================================================================
# Main
# ====================================================================================================

def main():

    for imgsz in IMGSZ_LIST:

        print("\n" + "=" * 100)
        print(
            f"Starting ID Switch experiment: "
            f"imgsz={imgsz}"
        )
        print("=" * 100)

        # -------------------------------------------------------------------------------------------
        # Tracker
        # -------------------------------------------------------------------------------------------

        tracker = ByteTracker(
            model_path=MODEL_PATH,
            device=DEVICE,
            conf=CONF,
            max_history=30,
        )

        # -------------------------------------------------------------------------------------------
        # Video
        # -------------------------------------------------------------------------------------------

        cap = cv2.VideoCapture(
            str(VIDEO_PATH)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {VIDEO_PATH}"
            )

        frame_width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        frame_height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        print(
            f"Video: "
            f"{frame_width}x{frame_height}"
        )

        print(
            f"Total frames: "
            f"{total_frames}"
        )

        # -------------------------------------------------------------------------------------------
        # Tracking State
        # -------------------------------------------------------------------------------------------

        previous_tracks = {}

        track_history = {}

        suspected_switches = []

        frame_count = 0

        # ===========================================================================================
        # Main Loop
        # ===========================================================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            result = tracker.track(
                frame,
                imgsz=imgsz,
            )

            tracks = result["tracks"]

            # ---------------------------------------------------------------------------------------
            # Current Frame Tracks
            # ---------------------------------------------------------------------------------------

            current_tracks = {}

            for track in tracks:

                track_id = int(
                    track["track_id"]
                )

                bbox = track["bbox"]

                confidence = track[
                    "confidence"
                ]

                current_tracks[track_id] = {
                    "bbox": bbox,
                    "confidence": confidence,
                }

                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append(
                    frame_count
                )

            # =======================================================================================
            # ID Switch Detection
            # =======================================================================================

            if previous_tracks:

                for old_id, old_track in previous_tracks.items():

                    old_box = old_track["bbox"]

                    # -------------------------------------------------------------------------------
                    # Old ID must disappear from current frame
                    # -------------------------------------------------------------------------------

                    if old_id in current_tracks:
                        continue

                    # -------------------------------------------------------------------------------
                    # Compare with every new ID
                    # -------------------------------------------------------------------------------

                    for new_id, new_track in current_tracks.items():

                        if new_id == old_id:
                            continue

                        new_box = new_track["bbox"]

                        iou = calculate_iou(
                            old_box,
                            new_box,
                        )

                        center_distance = (
                            calculate_center_distance(
                                old_box,
                                new_box,
                                frame_width,
                                frame_height,
                            )
                        )

                        # ---------------------------------------------------------------------------
                        # Possible ID Switch
                        # ---------------------------------------------------------------------------

                        if (
                            iou >= IOU_THRESHOLD
                            or center_distance
                            <= CENTER_DIST_THRESHOLD
                        ):

                            suspected_switches.append(
                                {
                                    "frame": frame_count,
                                    "old_id": old_id,
                                    "new_id": new_id,
                                    "iou": iou,
                                    "center_distance": center_distance,
                                    "confidence": new_track[
                                        "confidence"
                                    ],
                                }
                            )

                            print(
                                f"Frame {frame_count:<4} | "
                                f"Possible ID Switch: "
                                f"{old_id} -> {new_id} | "
                                f"IoU={iou:.3f} | "
                                f"Dist={center_distance:.3f} | "
                                f"Conf={new_track['confidence']:.3f}"
                            )

            # ---------------------------------------------------------------------------------------
            # Save Current Tracks
            # ---------------------------------------------------------------------------------------

            previous_tracks = current_tracks

            # ---------------------------------------------------------------------------------------
            # Progress
            # ---------------------------------------------------------------------------------------

            if frame_count % 50 == 0:

                print(
                    f"Processed: "
                    f"{frame_count}/{total_frames}"
                )

        cap.release()

        # ===========================================================================================
        # Filter Events
        # ===========================================================================================

        filtered_switches = []

        for event in suspected_switches:

            new_id = event["new_id"]

            history = track_history.get(
                new_id,
                [],
            )

            if len(history) >= MIN_NEW_ID_LENGTH:

                filtered_switches.append(
                    event
                )

        # ===========================================================================================
        # Final Results
        # ===========================================================================================

        print("\n")

        print("=" * 120)
        print("FINAL ID SWITCH ANALYSIS")
        print("=" * 120)

        print(
            f"Total suspected ID switch events: "
            f"{len(filtered_switches)}"
        )

        print("=" * 120)

        if not filtered_switches:

            print(
                "No suspected ID switch events detected."
            )

        else:

            print(
                f"{'Frame':<10}"
                f"{'OldID':<10}"
                f"{'NewID':<10}"
                f"{'IoU':<10}"
                f"{'Dist':<10}"
                f"{'Conf':<10}"
            )

            print("-" * 120)

            for event in filtered_switches:

                print(
                    f"{event['frame']:<10}"
                    f"{event['old_id']:<10}"
                    f"{event['new_id']:<10}"
                    f"{event['iou']:<10.3f}"
                    f"{event['center_distance']:<10.3f}"
                    f"{event['confidence']:<10.3f}"
                )

        print("=" * 120)


if __name__ == "__main__":
    main()