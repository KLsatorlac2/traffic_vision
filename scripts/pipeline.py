import cv2
from pathlib import Path

from tracking.tracker import ByteTracker


# ====================================================================================================
# Configuration
# ====================================================================================================

VIDEO_PATH = Path("datasets/videos/traffic.mp4")
OUTPUT_PATH = Path("runs/pipeline/traffic_counting.mp4")

MODEL_PATH = "yolo11n.pt"
DEVICE = 0

CONF = 0.15
IMGSZ = 1280
MAX_HISTORY = 30

LINE_A = ((320, 550), (570, 550))
LINE_B = ((590, 550), (840, 550))

LINE_TOLERANCE = 5


# ====================================================================================================
# Geometry
# ====================================================================================================

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def crossed_line(previous_y, current_y, line_y):
    return (
        previous_y < line_y - LINE_TOLERANCE
        and current_y >= line_y - LINE_TOLERANCE
    ) or (
        previous_y > line_y + LINE_TOLERANCE
        and current_y <= line_y + LINE_TOLERANCE
    )


def inside_line_x(center_x, line):
    x1 = min(line[0][0], line[1][0])
    x2 = max(line[0][0], line[1][0])
    return x1 <= center_x <= x2


# ====================================================================================================
# Drawing
# ====================================================================================================

def draw_line(frame, line, label):
    cv2.line(frame, line[0], line[1], (255, 0, 0), 3)

    x, y = line[0]

    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )


def draw_track(frame, track, history):
    x1, y1, x2, y2 = map(int, track["bbox"])

    track_id = track["track_id"]
    confidence = track["confidence"]

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"ID:{track_id} {confidence:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.circle(
        frame,
        (center_x, center_y),
        4,
        (0, 0, 255),
        -1,
    )

    for i in range(1, len(history)):
        cv2.line(
            frame,
            history[i - 1],
            history[i],
            (0, 0, 255),
            2,
        )


# ====================================================================================================
# Main
# ====================================================================================================

def main():

    print("=" * 100)
    print("YOLO + ByteTrack + DOUBLE LINE COUNTING PIPELINE")
    print("=" * 100)

    print(f"Model: {MODEL_PATH}")
    print(f"Confidence: {CONF}")
    print(f"Image size: {IMGSZ}")
    print(f"Line A: {LINE_A}")
    print(f"Line B: {LINE_B}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracker = ByteTracker(
        model_path=MODEL_PATH,
        device=DEVICE,
        conf=CONF,
        max_history=MAX_HISTORY,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create output video: {OUTPUT_PATH}"
        )

    line_a_y = LINE_A[0][1]
    line_b_y = LINE_B[0][1]

    previous_centers = {}

    # Each vehicle can only be counted once.
    counted_ids = set()

    direction_a = 0
    direction_b = 0

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = tracker.track(
            frame,
            imgsz=IMGSZ,
        )

        tracks = result["tracks"]

        for track in tracks:

            track_id = track["track_id"]
            bbox = track["bbox"]

            center_x, center_y = get_center(bbox)

            previous_y = previous_centers.get(track_id)

            # ----------------------------------------------------------------------------------------
            # Counting
            # ----------------------------------------------------------------------------------------

            if previous_y is not None and track_id not in counted_ids:

                crossed_a = (
                    inside_line_x(center_x, LINE_A)
                    and crossed_line(
                        previous_y,
                        center_y,
                        line_a_y,
                    )
                )

                crossed_b = (
                    inside_line_x(center_x, LINE_B)
                    and crossed_line(
                        previous_y,
                        center_y,
                        line_b_y,
                    )
                )

                if crossed_a:
                    direction_a += 1
                    counted_ids.add(track_id)

                    print(
                        f"Frame {frame_count}: "
                        f"ID {track_id} -> Direction A"
                    )

                elif crossed_b:
                    direction_b += 1
                    counted_ids.add(track_id)

                    print(
                        f"Frame {frame_count}: "
                        f"ID {track_id} -> Direction B"
                    )

            previous_centers[track_id] = center_y

            # ----------------------------------------------------------------------------------------
            # Draw tracking
            # ----------------------------------------------------------------------------------------

            history = tracker.get_history(
                track_id
            )

            draw_track(
                frame,
                track,
                history,
            )

        # --------------------------------------------------------------------------------------------
        # Draw counting lines
        # --------------------------------------------------------------------------------------------

        draw_line(
            frame,
            LINE_A,
            "LINE A",
        )

        draw_line(
            frame,
            LINE_B,
            "LINE B",
        )

        # --------------------------------------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------------------------------------

        total_count = direction_a + direction_b

        cv2.putText(
            frame,
            f"Direction A: {direction_a}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Direction B: {direction_b}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Total: {total_count}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        writer.write(frame)

        frame_count += 1

        if frame_count % 50 == 0:

            progress = (
                frame_count / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Processed: "
                f"{frame_count}/{total_frames} "
                f"({progress:.1f}%)"
            )

    cap.release()
    writer.release()

    # =================================================================================================
    # Final Results
    # =================================================================================================

    total_count = direction_a + direction_b

    print("\n" + "=" * 100)
    print("FINAL TRAFFIC COUNTING RESULTS")
    print("=" * 100)

    print(f"Direction A: {direction_a}")
    print(f"Direction B: {direction_b}")
    print(f"Total vehicles: {total_count}")
    print(f"Unique counted IDs: {len(counted_ids)}")
    print(f"Output: {OUTPUT_PATH}")

    print("=" * 100)


if __name__ == "__main__":
    main()