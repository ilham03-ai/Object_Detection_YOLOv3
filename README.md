# Object Detection with YOLOv3

Real-time object detection using YOLOv3 and OpenCV. Detects objects from the webcam and draws bounding boxes with labels.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- YOLOv3 weights (`yolov3.weights`) — not included in the repo, download from the official YOLO site

## Setup

```bash
pip install -r requirements.txt
```

Download `yolov3.weights` from [here](https://pjreddie.com/media/files/yolov3.weights) and put it in the root folder, then run:

```bash
python main.py
```

Press `q` to quit.

## How it works

Each frame from the webcam is passed through the YOLOv3 network. Detections with confidence > 50% are kept, then Non-Maximum Suppression filters out overlapping boxes. The result is drawn on screen with the class name and confidence score.

## What I Learned

- How to use OpenCV's DNN module to load and run a neural network without PyTorch or TensorFlow
- The importance of Non-Maximum Suppression — without it the same object gets detected multiple times with overlapping boxes
- How YOLO divides the image into a grid and outputs box coordinates as relative values (0 to 1), so you have to scale them back to the actual frame size
- Blob conversion: why the image needs to be normalized and resized before being fed into the network

## Files

- `main.py` — main script
- `yolov3.cfg` — network config
- `coco.names` — 80 COCO class names
