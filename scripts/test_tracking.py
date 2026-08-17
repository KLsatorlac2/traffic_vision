import cv2
from pathlib import Path

from tracking.tracker import ByteTracker


# ====================================================================================================
# Configuration
# ====================================================================================================

VIDEO_PATH = Path("datasets/videos/traffic.mp4")
OUTPUT_PATH = Path("runs/tracking/bytetrack_trajectory.mp4")

MODEL_PATH = "yolo11n.pt"
DEVICE = 0

CONF = 0.15
IMGSZ_LIST = [1280]

# ====================================================================================================
# Draw Track
# ====================================================================================================

def draw_track(frame, track, history):

    x1, y1, x2, y2 = map(
        int,
        track["bbox"],
    )

    track_id = track["track_id"]
    confidence = track["confidence"]

    label = f"ID:{track_id} {confidence:.2f}"

    # -----------------------------------------------------------------------------------------------
    # Bounding Box
    # -----------------------------------------------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    # -----------------------------------------------------------------------------------------------
    # Label
    # -----------------------------------------------------------------------------------------------

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # -----------------------------------------------------------------------------------------------
    # Trajectory
    # -----------------------------------------------------------------------------------------------

    for i in range(1, len(history)):

        x_prev, y_prev = history[i - 1]
        x_curr, y_curr = history[i]

        cv2.line(
            frame,
            (x_prev, y_prev),
            (x_curr, y_curr),
            (0, 0, 255),
            2,
        )


# ====================================================================================================
# Main
# ====================================================================================================

def main():

    results = []

    for imgsz in IMGSZ_LIST:

        print("\n" + "=" * 100)
        print(f"Starting experiment: imgsz={imgsz}")
        print("=" * 100)

        output_path = (
            Path("runs/tracking")
            / f"bytetrack_imgsz_{imgsz}.mp4"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tracker = ByteTracker(
            model_path=MODEL_PATH,
            device=DEVICE,
            conf=CONF,
            max_history=30,
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

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            result = tracker.track(
                frame,
                imgsz=imgsz,
            )

            tracks = result["tracks"]
            detections = result["detections"]

            for track in tracks:

                track_id = track["track_id"]

                history = tracker.get_history(
                    track_id
                )

                draw_track(
                    frame,
                    track,
                    history,
                )

            writer.write(frame)

            frame_count += 1

            if frame_count % 50 == 0:

                print(
                    f"Processed: "
                    f"{frame_count}"
                )

        cap.release()
        writer.release()

        # =================================================================================================
        # Statistics
        # =================================================================================================

        statistics = tracker.get_statistics()

        print("\nExperiment finished.")

        print(
            f"Total unique tracks: "
            f"{statistics['total_tracks']}"
        )

        print(
            f"Average track length: "
            f"{statistics['average_track_length']:.2f}"
        )

        print(
            f"Maximum track length: "
            f"{statistics['max_track_length']}"
        )

        print(
            f"Short tracks (<10 frames): "
            f"{statistics['short_tracks']}"
        )

        # ====================================================================================================
        # Track Lifetime / Lost Analysis
        # ====================================================================================================

        lifetimes = tracker.get_track_lifetimes()

        # ====================================================================================================
        # LOST GAP ANALYSIS
        # ====================================================================================================

        print("\n" + "=" * 120)
        print("TOP 10 TRACKS WITH LARGEST LOST GAPS")
        print("=" * 120)

        lost_candidates = [
            item
            for item in lifetimes
            if item["max_lost_gap"] > 0
        ]

        lost_candidates.sort(
            key=lambda x: x["max_lost_gap"],
            reverse=True,
        )

        print(
            f"{'ID':<8}"
            f"{'Start':<10}"
            f"{'End':<10}"
            f"{'Missing':<10}"
            f"{'MaxLost':<10}"
            f"{'AvgConf':<10}"
        )

        print("-" * 120)

        for item in lost_candidates[:10]:
            print(
                f"{item['track_id']:<8}"
                f"{item['start_frame']:<10}"
                f"{item['end_frame']:<10}"
                f"{item['missing_frames']:<10}"
                f"{item['max_lost_gap']:<10}"
                f"{item['average_confidence']:<10.3f}"
            )

        print("=" * 120)

        # ====================================================================================================
        # DETAILED LOST GAPS
        # ====================================================================================================

        print("\n" + "=" * 120)
        print("DETAILED LOST GAPS")
        print("=" * 120)

        for item in lost_candidates[:10]:

            print(
                f"\nTrack ID {item['track_id']} "
                f"| Start={item['start_frame']} "
                f"| End={item['end_frame']} "
                f"| Missing={item['missing_frames']} "
                f"| MaxLost={item['max_lost_gap']}"
            )

            if not item["lost_gaps"]:
                print("  No Lost gap.")
                continue

            for gap in item["lost_gaps"]:
                print(
                    f"  Lost: "
                    f"Frame {gap['start']} → {gap['end']} "
                    f"({gap['length']} frames)"
                )

        print("=" * 120)

if __name__ == "__main__":
    main()