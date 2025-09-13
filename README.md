# Video_Processing-_for-_Train-_Side_View
The aim of this project is to develop a video processing system that splits a full train video into images or frames of each coach, counts the number of coaches, organizes the output into structured folders, and get minimum images to identify full wagon .


Video Processing for Train Side View
Filename: video_split_and_detect.py

Overview:
This System
- Splits a full-side-view train video into per-coach video clips by detecting gaps between coaches.
- Counts coaches and creates per-coach folders named <train_number>_<counter>.
- Extracts frames from each coach clip and saves them as <train_number>_<counter>_<frame_number>.jpg.
- Runs component detection (Door / Door Open / Door Closed) on extracted frames.

Usage (example):
python video_split_and_detect.py \
  --input DHN-upper-side-view-2025-08-31-11-28-15-377.mp4 \
  --train-number 12309 \
  --outdir ./output \
  --min_gap_area_percent 1.5

Notes:
- The script first uses background subtraction to compute a foreground area per frame.
- It finds temporal minima (gaps between coaches) on that area signal to split the video into segments.
- For detection, it will try to use a YOLOv5/YOLOv8 model if torch and a weights file are available at "yolov5s.pt" or "best.pt".
  If not available, it falls back to a simple contour-based heuristic (rectangle detection) to propose door candidates.

Caveats:
- This is a practical, general-purpose pipeline. Depending on your camera angle, lighting, and train speed you will likely
  need to tune the smoothing window and gap thresholds in the command-line options.
- For production-grade door open/closed detection, use a labelled dataset and train a dedicated detector.
