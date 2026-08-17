import cv2
import time
import torch
from pathlib import Path

from detection.detector import VehicleDetector


# ====================================================================================================
# Configuration
# ====================================================================================================

VIDEO_PATH = Path("datasets/videos/traffic.mp4")
OUTPUT_DIR = Path("runs/detection")

MODEL_PATH = "yolo11n.pt"
DEVICE = 0

CONF = 0.15
IMGSZ_LIST = [640, 960, 1280]


# ====================================================================================================
# Single Experiment
# ====================================================================================================

def run_experiment(imgsz):
    """Run one YOLO image-size experiment and return the statistics."""

    print("\n" + "=" * 100)
    print(f"Starting experiment: imgsz={imgsz}, conf={CONF}")
    print("=" * 100)

    # ------------------------------------------------------------------------------------------------
    # Output Path
    # ------------------------------------------------------------------------------------------------

    output_path = OUTPUT_DIR / f"imgsz_{imgsz}.mp4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------------
    # Create Detector
    # ------------------------------------------------------------------------------------------------

    detector = VehicleDetector(
        model_path=MODEL_PATH,
        device=DEVICE,
        conf=CONF,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------------------------------------
    # Open Video
    # ------------------------------------------------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total frames: {total_frames}")

    # ------------------------------------------------------------------------------------------------
    # Video Writer
    # ------------------------------------------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        video_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # ------------------------------------------------------------------------------------------------
    # Statistics Initialization
    # ------------------------------------------------------------------------------------------------

    frame_count = 0
    total_detections = 0
    total_inference_time = 0.0

    start_time = time.perf_counter()

    # =================================================================================================
    # Main Video Processing Loop
    # =================================================================================================

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # YOLO inference
        inference_start = time.perf_counter()
        detections = detector.predict(frame, imgsz=imgsz)
        inference_end = time.perf_counter()

        total_inference_time += inference_end - inference_start
        total_detections += len(detections)

        # Draw detection results
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(frame)
        frame_count += 1

        # Progress
        if frame_count % 50 == 0:
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

    # =================================================================================================
    # Release Resources
    # =================================================================================================

    cap.release()
    writer.release()

    # =================================================================================================
    # Calculate Statistics
    # =================================================================================================

    total_time = time.perf_counter() - start_time

    average_detections = total_detections / frame_count if frame_count > 0 else 0
    actual_fps = frame_count / total_time if total_time > 0 else 0
    average_inference_ms = total_inference_time / frame_count * 1000 if frame_count > 0 else 0

    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        gpu_memory_mb = 0

    # ------------------------------------------------------------------------------------------------
    # Experiment Result
    # ------------------------------------------------------------------------------------------------

    result = {
        "imgsz": imgsz,
        "conf": CONF,
        "frames": frame_count,
        "detections": total_detections,
        "avg_det_per_frame": average_detections,
        "total_time": total_time,
        "actual_fps": actual_fps,
        "avg_inference_ms": average_inference_ms,
        "gpu_memory_mb": gpu_memory_mb,
        "output": str(output_path),
    }

    print("\nExperiment finished.")
    print(f"Average detections/frame: {average_detections:.2f}")
    print(f"Average inference time: {average_inference_ms:.2f} ms")
    print(f"Actual processing FPS: {actual_fps:.2f}")
    print(f"Peak GPU memory: {gpu_memory_mb:.0f} MB")
    print(f"Total time: {total_time:.2f} s")
    print(f"Output: {output_path}")

    return result


# ====================================================================================================
# Main
# ====================================================================================================

def main():
    print("=" * 100)
    print("YOLO Image Size Experiment")
    print("=" * 100)

    print(f"Model: {MODEL_PATH}")
    print(f"Confidence: {CONF}")
    print(f"Image sizes: {IMGSZ_LIST}")

    results = []

    # ------------------------------------------------------------------------------------------------
    # Run All Experiments
    # ------------------------------------------------------------------------------------------------

    for imgsz in IMGSZ_LIST:
        result = run_experiment(imgsz)
        results.append(result)

    # =================================================================================================
    # Final Experiment Summary
    # =================================================================================================

    print("\n")
    print("=" * 110)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 110)

    print(
        f"{'ImgSz':<10}"
        f"{'AvgDet':<15}"
        f"{'Infer(ms)':<15}"
        f"{'FPS':<12}"
        f"{'GPU(MB)':<12}"
        f"{'Time(s)':<12}"
    )

    print("-" * 110)

    for result in results:
        print(
            f"{result['imgsz']:<10}"
            f"{result['avg_det_per_frame']:<15.2f}"
            f"{result['avg_inference_ms']:<15.2f}"
            f"{result['actual_fps']:<12.2f}"
            f"{result['gpu_memory_mb']:<12.0f}"
            f"{result['total_time']:<12.2f}"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()