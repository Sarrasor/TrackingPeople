# Tracking people

![Results of tracking]('images/result.png')

![Results of tracking]('images/result2.png')

Here is the link to folder with some results:
[Click](https://drive.google.com/open?id=1bFf2Bz0xscJkE7tbkzITag0BEhVA9H0z)

## Repository contents description
- `input` - folder for input videos
- `models` - folder for models
- `output` - folder for output videos
- `people_tracker` - folder with code
	- `yolo` - YOLOv3 model
	- `detector.py` - class for running detectors from TF Zoo
	- `detector_yolo.py` - class for running YOLOv3 detector
	- `draw_detections.py` - code to test detections
	- `main.py` - track people using Bounding Box Kalman Tracker
	- `main_center.py` - track people using Centroid Kalman Tracker
	- `tools.py` - helper functions
	- `tracker.py` - Bounding Box Kalman Tracker
	- `tracker_center.py` - Centroid Kalman Tracker
 - `Dockerfile`
- `environment.yml`

## Problem description
We have a `video.mp4` 

The task is to find all people in each frame, draw bounding box, and assign an ID to each person. The previously detected person should have the same ID when detected again. A new person should get an ID incrementally.

We should save the output in `output.mp4` 

We make an assumption that there will be no instantaneous camera movements and that objects cannot dissapear and appear instantaneously.

## Solution approach
The aproach is the following:

1) Generate list of bounding box detections with some Neural Network
2) Solve data association problem
2) Track boxes using Kalman filter. Detections are used as measurements for Kalman filter

## Detection
Let's use a Neural Network that can detect objects to generate bounding boxes around people.

The selection of a Neural Network for detection highly influences the results of tracking. For example, if we use `ssd_mobilenet_v2_coco`, the system will work in real time, but the results are bad due to frequent bounding box jitter.

Also, the dataset matters a lot. For example, `faster_rcnn_nas` that was trained on COCO dataset has worse detection accuracy than the faster `faster_rcnn_resnet101_kitti` model that was trained on KITTI dataset.

Out of all tested models, YOLOv3 has shown the best result in terms of speed/accuracy. On the system with GPU tracking with YOLOv3 should work in real time.

## Data association
The data association problem is approached as an optimization problem. The task is to minimize weighted sum of bounding boxes. Cost matrix for the centroid tracking approach was constructed using eucledian distance between centroids. IOU metric was used for the bounding box approach.

The optimization problem is solved with Hungarian algorithm (also known as Munkres algorithm).

## Tracking
Given the associated data as measurements we can track bounding boxes. Two approaches are tested: 

### Centroid tracking
The state is represented as coordinates of the bounding box centroid. The centroid of the associated measurement is calculated and passed as a measurement to Kalman filter

This approach seems to be more robust to bad detections

### Bounding box
The state is represented as coordinates of corners of the bounding box. The associated measurement is passed as a measurement to Kalman filter

This approach seems to be more robust to occlusions

## How to improve the soution
- Do proper metapatameter tuning (Kalman, Neural Network, thresholds)
- Retrain models on humans using transfer learning
- Select better model for Kalman filter
- Combine Kalman filter with more advanced trackers like GOTURN
- Solve data association problem using additional methods: 
	- Bipartite Graphs approaches
	- Dynamic models integration
	- etc
- Do thorough research on new detection and tracking methods

## Relation to code
You can run centroid tracking approach using:
`python3 main_center.py ../input/video.mp4`

You can run bounding box tracking approach using:
`python3 main.py ../input/video.mp4`

## Hardware
`Intel Core i5-6200U CPU @ 2.30GHz × 4`

## Software stack
- **Python**
	- `numpy v1.18.2`
	- `scipy v1.4.1`
- **OpenCV** (`cv2 v4.2`)
- **TensorFlow** `(tensorflow v1.14.0)`

## How to run code
`cd` to the `TrackingPeople` folder first. Then, download your models:

### TensorFlow Zoo
[Donwnload model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Unpack downloaded model to: 

`models`

Don't forget to select the downloaded model in `main.py` or `main_center.py` 

### YOLO
[Download weights](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)
[Download weights mirror](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing)

Save downloaded files to:

`people_tracker/yolo/data/darknet_weights/` directory

Don't forget to select the downloaded model in `main.py` or `main_center.py` 

### Docker
There is a Docker environment ready for execution.

Run Docker compose:
```
sudo xhost +local:root 
docker-compose -f environment.yml up --build
```

Enter the Docker container:
```
docker exec -it trackingpeople_tracker_1 bash
```

There you can run some code. Here is an example:

```
cd people_tracker
python3 main_center.py ../input/video.mp4
```

## Many thanks to:
[TF model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
[Yolo](https://github.com/wizyoung/YOLOv3_TensorFlow)
[Sort](https://github.com/abewley/sort)