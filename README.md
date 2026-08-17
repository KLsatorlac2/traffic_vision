# Traffic Vision

YOLO11 + ByteTrack based traffic vehicle detection, multi-object tracking and bidirectional vehicle counting system.

## Demo

![Traffic Vision Demo](assets/demo.gif)

## 1. Project Overview

This project implements a complete traffic vision pipeline based on YOLO11 and ByteTrack.

The system performs:

* Vehicle detection
* Multi-object tracking
* Track ID assignment
* Vehicle trajectory visualization
* Bidirectional vehicle counting
* Detection and tracking experiments
* Tracking lifetime analysis
* Lost gap analysis

The final pipeline processes a traffic monitoring video and counts vehicles passing through two manually defined counting lines.

---

## 2. Pipeline

```text
Input Video
     │
     ▼
 YOLO11 Detection
     │
     ▼
 ByteTrack
     │
     ▼
 Track ID
     │
     ▼
 Vehicle Center Point
     │
     ▼
 Double Line Crossing
     │
     ▼
 Direction Classification
     │
     ▼
 Vehicle Counting
     │
     ▼
 Output Video
```

---

## 3. Main Features

### 3.1 Vehicle Detection

YOLO11 is used as the object detector.

The project evaluates different:

* Confidence thresholds
* Input image sizes

### 3.2 Multi-Object Tracking

ByteTrack is used to associate vehicle detections between consecutive frames.

The tracking parameters were experimentally evaluated, especially:

```text
match_thresh
```

### 3.3 Bidirectional Vehicle Counting

Two counting lines are manually defined according to the road geometry:

```python
LINE_A = ((320, 550), (570, 550))
LINE_B = ((590, 550), (840, 550))
```

The vehicle center point is calculated as:

```python
center_y = (y1 + y2) / 2
```

A vehicle is counted when its center point crosses the corresponding counting line.

---

## 4. Environment

Tested environment:

```text
OS: Windows 10
Python: 3.11.14
CPU: Intel Core i7-13700H
RAM: 15.6 GB
GPU: NVIDIA GeForce RTX 4060 Laptop GPU 8 GB
PyTorch: 2.5.0+cu118
CUDA: 11.8
Ultralytics: 8.3.163
```

---

## 5. Installation

Create the Conda environment:

```bash
conda create -n traffic_vision python=3.11
conda activate traffic_vision
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify the PyTorch environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected result:

```text
2.5.0+cu118
True
```

Verify Ultralytics:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

---

## 6. Project Structure

```text
traffic_vision/
│
├── analytics/
│
├── configs/
│   ├── bytetrack_custom.yaml
│   └── pipeline.yaml
│
├── datasets/
│   └── videos/
│       └── traffic.mp4
│
├── detection/
│   └── detector.py
│
├── tracking/
│   └── tracker.py
│
├── scripts/
│   ├── detect_video.py
│   ├── test_tracking.py
│   ├── analyze_id_switch.py
│   ├── analyze_lost_gaps.py
│   └── pipeline.py
│
├── runs/
│   ├── detection/
│   ├── tracking/
│   ├── lost_analysis/
│   └── pipeline/
│
├── yolo11n.pt
├── requirements.txt
└── README.md
```

---

## 7. Quick Start

Place the traffic video at:

```text
datasets/videos/traffic.mp4
```

Run the final pipeline:

```bash
python -m scripts.pipeline
```

The output video will be saved to:

```text
runs/pipeline/traffic_counting.mp4
```

---

# 8. Experiments

## 8.1 Confidence Threshold Experiment

The following confidence thresholds were evaluated:

```text
0.25
0.15
0.10
```

### Results

| Confidence | Avg Detection / Frame | Inference Time (ms) |   FPS |
| ---------: | --------------------: | ------------------: | ----: |
|       0.25 |                 14.27 |               17.26 | 41.07 |
|       0.15 |                 21.18 |                9.45 | 58.51 |
|       0.10 |                 27.96 |               11.45 | 54.04 |

Lowering the confidence threshold increased the number of detected vehicles.

However, lower confidence thresholds also introduced more false detections.

A confidence threshold of:

```text
0.15
```

was selected for the final pipeline as a balance between detection recall and false detections.

---

## 8.2 Image Size Experiment

Three input image sizes were evaluated:

```text
640
960
1280
```

### Results

| Image Size | Avg Detection / Frame | Inference Time (ms) |   FPS | GPU (MB) |
| ---------: | --------------------: | ------------------: | ----: | -------: |
|        640 |                 21.18 |               17.76 | 39.94 |       69 |
|        960 |                 31.10 |               13.75 | 46.88 |      114 |
|       1280 |                 46.94 |               15.31 | 43.50 |      171 |

Increasing the input image size significantly improved the detection of distant vehicles.

The `1280` configuration recovered many distant vehicles that were missed at `640` and `960`.

The additional detections were mostly real vehicles.

Therefore:

```text
imgsz = 1280
```

was selected for the final pipeline.

---

## 8.3 ByteTrack Experiment

Different ByteTrack matching thresholds were evaluated.

The original lower matching threshold resulted in frequent ID switches.

Increasing:

```text
match_thresh
```

significantly improved ID stability for nearby and middle-distance vehicles.

The final configuration uses:

```text
match_thresh = 0.85
```

---

## 8.4 Tracking Experiment

The tracking performance was compared using different input image sizes.

### Results

| Image Size | Tracks | Avg Track Length | Track Max | Short Tracks (<10) |
| ---------: | -----: | ---------------: | --------: | -----------------: |
|        640 |     61 |            56.15 |       233 |                 13 |
|        960 |     71 |            73.17 |       233 |                 10 |
|       1280 |     97 |            80.79 |       233 |                 20 |

Increasing image size improved the ability to maintain tracks for distant vehicles.

Although some distant vehicles still failed to obtain stable IDs, the overall tracking performance improved substantially.

---

## 8.5 Track Lifetime Analysis

The project records:

* Track start frame
* Track end frame
* Track lifetime
* Missing frames
* Maximum lost gap
* Average detection confidence

This allows tracking failures to be analyzed quantitatively instead of relying only on visual inspection.

---

## 8.6 Lost Gap Analysis

Lost gaps were analyzed to distinguish between:

```text
Detection failure
```

and:

```text
Association failure
```

The analysis showed that many long tracking gaps were caused by the detector failing to produce a nearby detection rather than ByteTrack incorrectly associating two detections.

This indicates that improving the detector, especially for small and partially occluded vehicles, would likely provide more benefit than continuously increasing the tracking threshold.

---

# 9. Final Configuration

The final pipeline uses:

| Component        | Configuration |
| ---------------- | ------------- |
| Model            | YOLO11n       |
| Confidence       | 0.15          |
| Input Image Size | 1280          |
| Tracker          | ByteTrack     |
| Match Threshold  | 0.85          |

### Counting Lines

```python
LINE_A = ((320, 550), (570, 550))
LINE_B = ((590, 550), (840, 550))
```

The counting point is calculated using:

```python
center_y = (y1 + y2) / 2
```

A vehicle is counted when its center point crosses the corresponding counting line.

---

# 10. Final Counting Example

Example output:

```text
========================================
FINAL TRAFFIC COUNTING RESULTS
========================================

Direction A: 5
Direction B: 19
Total vehicles: 24
Unique counted IDs: 24

Output:
runs/pipeline/traffic_counting.mp4
```

The final pipeline successfully integrates:

```text
Detection
    +
Tracking
    +
Trajectory
    +
Line Crossing
    +
Direction Classification
    +
Vehicle Counting
```

---

# 11. Known Limitations

The current video presents several challenges.

### 11.1 Small Objects

Vehicles become significantly smaller toward the distant part of the bridge.

This makes detection more difficult.

### 11.2 Occlusion

Traffic density is relatively high and vehicles are sometimes close to each other.

This can cause:

* Missing detections
* Track loss
* ID instability

### 11.3 Shadows

The video was recorded near dusk and many vehicles cast strong shadows.

These shadows can increase detection difficulty.

### 11.4 Camera Motion

The camera is not completely static.

Camera motion analysis showed measurable frame-to-frame global motion.

Camera motion compensation was investigated but was not included in the final pipeline.

### 11.5 Perspective

The bridge is viewed from an oblique aerial perspective.

Vehicles become smaller toward the distant part of the image, increasing the difficulty of detection and tracking.

---

# 12. Future Improvements

Possible future improvements include:

* Using a larger YOLO model
* Training on a traffic-specific dataset
* Improving small-object detection
* Using segmentation-based vehicle detection
* Improving camera motion compensation
* Using stronger tracking algorithms such as BoT-SORT
* Adding vehicle type classification
* Adding speed estimation
* Adding automatic counting-line placement
* Deploying the system as a real-time application

---

# 13. Conclusion

This project demonstrates a complete traffic analysis pipeline based on YOLO11 and ByteTrack.

Through systematic experiments on detection confidence, image size, and tracking parameters, the final configuration was selected as:

```text
YOLO11n
conf = 0.15
imgsz = 1280
ByteTrack
match_thresh = 0.85
```

The final system can detect and track vehicles and perform bidirectional traffic counting using manually defined counting lines.

The project also includes quantitative analysis of tracking lifetime, lost gaps, and detection/association failures, providing a basis for further optimization and future system improvements.
