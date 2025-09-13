"""
Video Processing for Train Bottom View
Filename: video_split_and_detect.py

Overview:
- Splits a full-side-view train video into per-coach video clips by detecting gaps between coaches.
- Counts coaches and creates per-coach folders named <train_number>_<counter>.
- Extracts frames from each coach clip and saves them as <train_number>_<counter>_<frame_number>.jpg.
- Runs component detection (Door / Door Open / Door Closed) on extracted frames.

Usage (example):
python video_split_and_detect.py \
  --input /mnt/data/DHN-upper-side-view-2025-08-31-11-28-15-377.mp4 \
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

"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import math
import torch

import cv2
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Try importing torch and a YOLO utility if available - optional
print("started executing script")
try:
    
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def compute_foreground_area_signal(video_path, resize_width=None, step=1, history=500, varThreshold=16):
    """
    Returns a list of foreground pixel counts per analyzed frame and fps/duration info.
    We use cv2.createBackgroundSubtractorMOG2 to get a stable foreground mask.
    Parameters:
      - resize_width: if set, frames are resized to this width for faster processing (maintains aspect ratio)
      - step: analyze every 'step' frame
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=False)

    areas = []
    times = []
    frame_idx = 0
    resized_dim = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        if resize_width is not None:
            if resized_dim is None:
                scale = resize_width / w
                resized_dim = (resize_width, int(h * scale))
            frame_r = cv2.resize(frame, resized_dim)
        else:
            frame_r = frame

        gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        fg = bg.apply(gray)
        # clean
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        area = int(np.count_nonzero(fg))
        areas.append(area)
        times.append(frame_idx / fps)

        frame_idx += 1

    cap.release()
    return np.array(areas), np.array(times), fps, total_frames


def smooth_signal(sig, window=31):
    # simple moving average smoothing
    if window <= 1:
        return sig
    window = min(window, len(sig) if len(sig)%2==1 else len(sig)-1)
    if window < 3:
        return sig
    kernel = np.ones(window) / window
    padded = np.pad(sig, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def find_split_times_from_area(areas, times, min_gap_factor=0.75, min_gap_area_percent=1.0, smooth_window=31):
    """
    Detect gaps (minima) in the foreground area time-series. Returns list of split times (seconds).
    Strategy:
      - smooth the area signal
      - find local minima where the smoothed area drops below a threshold (percent of max)
      - return times where minima occur
    """
    if len(areas) == 0:
        return []

    smooth = smooth_signal(areas.astype(float), smooth_window)
    # threshold: minima below percentile
    max_area = smooth.max()
    threshold = max_area * (min_gap_area_percent / 100.0)

    # find indices where smooth < threshold (candidate gaps)
    candidates = np.where(smooth < threshold)[0]
    if len(candidates) == 0:
        # fallback: use valley detection using relative minima
        from scipy.signal import argrelextrema
        minima = argrelextrema(smooth, np.less, order=max(1, len(smooth)//50))[0]
        return list(times[minima])

    # group contiguous candidate indices into gap regions and choose center time
    groups = []
    current = [candidates[0]]
    for idx in candidates[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    split_times = []
    for g in groups:
        center_idx = g[len(g)//2]
        split_times.append(times[center_idx])

    # remove edges if too close to start/end
    split_times = [t for t in split_times if t > 0.2 * times[-1] * 0 + 0 and t < times[-1] - 0.1]
    return split_times


def make_segments_from_splits(split_times, total_duration):
    """
    Given split times (times where gaps occur) return list of (start, end) segments assumed to be coaches.
    E.g. split_times = [2.5, 5.0] and duration=10 => segments: [0->2.5, 2.5->5.0, 5.0->10]
    """
    times = [0.0] + sorted(split_times) + [total_duration]
    segments = []
    for i in range(len(times)-1):
        s = times[i]
        e = times[i+1]
        if e - s > 0.1:  # ignore vanishing segments
            segments.append((s, e))
    return segments


def write_segment_clip(input_video, start_s, end_s, out_path):
    """Uses moviepy's ffmpeg_extract_subclip to write the subclip."""
    ffmpeg_extract_subclip(str(input_video), start_s, end_s, str(out_path))



def extract_frames_from_video(video_path, out_dir, train_number, coach_idx, every_n_frames=1):
    """
    Save frames from video_path into out_dir with naming <train_number>_<coach_idx>_<frame_number>.jpg
    Returns number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip {video_path}")
    ensure_dir(Path(out_dir))
    saved = 0
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_no % every_n_frames != 0:
            continue
        fname = f"{train_number}_{coach_idx}_{saved+1}.jpg"
        cv2.imwrite(str(Path(out_dir) / fname), frame)
        saved += 1
    cap.release()
    return saved


# ---------- Simple heuristic detector (fallback) ----------

def detect_doors_contour_based(image):
    """
    Heuristic: find rectangular-ish contours in the middle-lower vertical band of the coach image.
    Returns list of dicts: {label: 'Door'/'Door_Open'/'Door_Closed', bbox: (x,y,w,h), score:0.5}
    This is a weak heuristic intended as a fallback when no trained detector is available.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        area = wc*hc
        if area < 0.002 * (w*h):
            continue
        aspect = hc / (wc + 1e-6)
        # plausible door shapes are tall vertical rectangles
        if 1.0 < aspect < 5.0 and wc > w*0.03 and hc > h*0.15:
            # crude open/closed heuristic: if interior (center vertical strip) has lots of dark pixels -> closed
            cx = x + wc//2
            left = max(x + int(wc*0.1), 0)
            right = min(x + int(wc*0.9), w)
            center_strip = gray[y+int(hc*0.2):y+int(hc*0.8), left:right]
            mean_brightness = float(np.mean(center_strip))
            label = 'Door_Closed' if mean_brightness < 120 else 'Door_Open'
            results.append({'label': label.replace('_',' '), 'bbox': (x,y,wc,hc), 'score': 0.4})
    return results


# ---------- YOLO detection wrapper (optional) ----------

def detect_with_yolo(image, model):
    """
    Run a YOLO model (if supplied) to detect doors. Expects the model to follow the ultralytics/torch.hub signature
    Returns list of dicts {label, bbox, score}
    """
    # model should be a torch.hub loaded model or ultralytics yolo model
    try:
        results = model(image)
        # results may be a custom object - try to read .xyxy[0] or .pred
        detected = []
        if hasattr(results, 'xyxy'):
            # yolov5 style
            preds = results.xyxy[0].cpu().numpy()
            names = model.names if hasattr(model, 'names') else {0:'obj'}
            for *xyxy, conf, cls in preds:
                x1,y1,x2,y2 = map(int, xyxy)
                label = names.get(int(cls), str(int(cls)))
                detected.append({'label': label, 'bbox': (x1,y1,x2-x1,y2-y1), 'score': float(conf)})
        elif hasattr(results, 'pred'):  # ultralytics v8 style
            preds = results.pred[0].cpu().numpy()
            names = model.names if hasattr(model, 'names') else {}
            for *xyxy, conf, cls in preds:
                x1,y1,x2,y2 = map(int, xyxy[:4])
                detected.append({'label': names.get(int(cls), str(int(cls))), 'bbox': (x1,y1,x2-x1,y2-y1), 'score': float(conf)})
        else:
            # fallback: try results as list of dicts
            for det in results:
                detected.append(det)
        return detected
    except Exception as e:
        print('YOLO detection error:', e)
        return []


def annotate_and_save(image_path, detections, out_path):
    img = cv2.imread(str(image_path))
    for d in detections:
        x,y,w,h = map(int, d['bbox'])
        label = d.get('label','obj')
        score = d.get('score',0)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        text = f"{label} {score:.2f}" if score else label
        cv2.putText(img, text, (x, max(y-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imwrite(str(out_path), img)


def process_coach_folder(folder_path, train_number, coach_idx, yolo_model=None):
    """
    For each jpg in folder, run detection (YOLO if available, else heuristic), annotate and overwrite.
    """
    folder = Path(folder_path)
    jpgs = sorted(folder.glob(f"{train_number}_{coach_idx}_*.jpg"))
    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        detections = []
        if yolo_model is not None:
            detections = detect_with_yolo(img, yolo_model)
        if not detections:
            detections = detect_doors_contour_based(img)
        annotate_and_save(jpg, detections, jpg)


def main():
    parser = argparse.ArgumentParser(description='Split train video into coach clips, extract frames, detect doors')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--train-number', required=True, help='Train number code to use in filenames')
    parser.add_argument('--outdir', default='./output', help='Output directory')
    parser.add_argument('--resize-width', type=int, default=640, help='Resize width for analysis speed')
    parser.add_argument('--step', type=int, default=1, help='Analyze every Nth frame')
    parser.add_argument('--min_gap_area_percent', type=float, default=1.5,
                        help='Percent of max foreground area below which we consider a gap (lower = fewer splits)')
    parser.add_argument('--smooth-window', type=int, default=31, help='Smoothing window for area signal')
    parser.add_argument('--every-nth-frame', type=int, default=10, help='Save every Nth frame from clips')
    parser.add_argument('--yolo-weights', default=None, help='Optional YOLO weights file (yolov5s.pt or best.pt)')
    args = parser.parse_args()

    input_video = Path(args.input)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print('Computing foreground area signal...')
    areas, times, fps, total_frames = compute_foreground_area_signal(input_video, resize_width=args.resize_width, step=args.step)
    duration = total_frames / (fps or 25.0)
    print(f'Video FPS {fps:.2f}, frames {total_frames}, approx duration {duration:.2f}s')

    print('Detecting gap times...')
    try:
        splits = find_split_times_from_area(areas, times, min_gap_area_percent=args.min_gap_area_percent, smooth_window=args.smooth_window)
    except Exception as e:
        print('Error detecting splits, fallback to no-splits:', e)
        splits = []

    segments = make_segments_from_splits(splits, duration)
    print(f'Detected {len(segments)} segments (potential coaches)')

    # Load YOLO model if requested/available
    yolo_model = None
    if args.yolo_weights and TORCH_AVAILABLE:
        try:
            print('Loading YOLO model', args.yolo_weights)
            # try ultralytics hub API or torch.hub
            if 'yolov5' in args.yolo_weights.lower() or args.yolo_weights.endswith('.pt'):
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.yolo_weights, force_reload=False)
            else:
                # try to load with ultralytics
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            yolo_model = model
        except Exception as e:
            print('Failed to load YOLO model:', e)
            yolo_model = None

    # For each segment, save clip, extract frames, run detection
    for idx, (s,e) in enumerate(segments, start=1):
        coach_folder = outdir / f"{args.train_number}_{idx}"
        ensure_dir(coach_folder)
        clip_path = coach_folder / f"{args.train_number}_{idx}.mp4"
        print(f'Writing coach {idx}: {s:.2f}s -> {e:.2f}s to {clip_path}')
        try:
            write_segment_clip(input_video, s, e, clip_path)
        except Exception as ex:
            print('ffmpeg clip error:', ex)
            continue

        print('Extracting frames...')
        n_saved = extract_frames_from_video(clip_path, coach_folder, args.train_number, idx, every_n_frames=args.every_nth_frame)
        print(f'Saved {n_saved} frames for coach {idx}')

        print('Running detection and annotating frames...')
        process_coach_folder(coach_folder, args.train_number, idx, yolo_model=yolo_model)

    print('Finished. Output saved to', outdir)


if __name__ == '__main__':
    main()
