"""
Track people using centroids

Attributes:
    DETECT_FREQ (int): Description
    DISTANCE_THRESH (int): maximum eucledian distance
    MAX_MISSES (int): number unmatched detections before tracker deletion
    MIN_HITS (int): number matched detections before tracker creation
    next_id (int): next available id
    persons_list (list): list of person trackers
    SAVE_VIDEO (bool): set as True if you need to save the video
    SHOW_SCALE (float): scale of the displayed video
"""
import numpy as np
import cv2
import sys
from scipy.optimize import linear_sum_assignment
import time

import tools
import detector
import detector_yolo
import tracker_center

# Whether to save output video or no
SAVE_VIDEO = True

# Number of consecutive unmatched detections before tracker deletion
MAX_MISSES = 10

# Number of consecutive matched detections before tracker creation
MIN_HITS = 2

# Output video scale
SHOW_SCALE = 0.5

# Maximum eucledian distance between
# detection and tracker to be considered as a match
DISTANCE_THRESH = 100

# Detection frequency
DETECT_FREQ = 1

persons_list = []
next_id = 0


def assign_detections_to_trackers(trackers, detections, dst_thrd=100):
    """
    Solves data association problem

    Args:
        trackers (list): list of trackers' bounding boxes
        detections (list): list of detections' bounding boxes
        dst_thrd (float, optional): maximum distance btw matched box centers

    Returns:
        (list, list, list): matches, new detections, unmatched trackers
    """

    # Create cost matrix
    eucledian_costs = np.zeros(
        (len(trackers), len(detections)), dtype=np.float32)

    # Fill cost matrix using Eucledian Distance metric
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            detection = [det[0] + int((det[2] - det[0]) / 2),
                         det[1] + int((det[3] - det[1]) / 2)]
            tracker = [trk[0] + int((trk[2] - trk[0]) / 2),
                       trk[1] + int((trk[3] - trk[1]) / 2)]
            eucledian_costs[t, d] = tools.eucledian(tracker, detection)

    # Solve data association problem
    matched_idx = linear_sum_assignment(eucledian_costs)
    matched_idx = np.asanyarray(matched_idx).T

    unmatched_trackers, unmatched_detections = [], []

    # Save trackers without new detections
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    # Save new detections that have no tracker
    for d, det in enumerate(detections):
        if(d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # Check the "safe" distance between detection and tracker to be less
    # than dst_trhd. Append tracker to unmatched trackers otherwise
    for m in matched_idx:
        if(eucledian_costs[m[0], m[1]] > dst_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    # Prepare matches
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def predict(img, start_time):
    """
    Tracks people without detection

    Args:
        img (np.array): current frame
        start_time (float): time when the current frame was captured

    Returns:
        np: current frame with bounding boxes
    """
    global persons_list

    # Predict state for each tracker without measurement
    for person in persons_list:
        person.predict_state(time.time() - start_time)

        x = person.x.T[0].tolist()
        x = [x[0] - int(person.height / 2), x[1] - int(person.width / 2),
             x[0] + int(person.height / 2), x[1] + int(person.width / 2)]
        person.box = x

        x = person.x.T[0].tolist()[0]
        y = person.x.T[0].tolist()[1]
        img = cv2.circle(img, (y, x), 4, (0, 255, 0), -1)
        img = tools.draw_box(person.id, img, person.box)

    return img


def track(img, start_time):
    """
    Tracks people with detection

    Args:
        img (np.array): current frame
        start_time (float): time when the current frame was captured

    Returns:
        np: current frame with bounding boxes
    """

    global persons_list
    global next_id

    # Get detections
    detections = det.get_detections(img)

    # Create list of existing persons' boxes
    persons_boxes = []
    if len(persons_list) > 0:
        for person in persons_list:
            persons_boxes.append(person.box)

    # Solve data association problem
    matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(persons_boxes, detections, dst_thrd=DISTANCE_THRESH)

    # Update trackers with measurements
    if matched.size > 0:
        for person_idx, det_idx in matched:
            # Extract measurement
            z = detections[det_idx]
            center = [z[0] + int((z[2] - z[0]) / 2),
                      z[1] + int((z[3] - z[1]) / 2)]

            img = cv2.circle(img, (center[1], center[0]), 6, (0, 0, 255), -1)
            center = np.expand_dims(center, axis=0).T

            # Extract person tracker
            person = persons_list[person_idx]
            person.hits += 1
            person.misses = 0

            person.process_measurement(center, time.time() - start_time)
            person.width = z[3] - z[1]
            person.height = z[2] - z[0]
            x = person.x.T[0].tolist()
            x = [x[0] - int(person.height / 2), x[1] - int(person.width / 2),
                 x[0] + int(person.height / 2), x[1] + int(person.width / 2)]
            persons_boxes[person_idx] = x
            person.box = x

    # Update trackers without measurements
    if len(unmatched_trks) > 0:
        for person_idx in unmatched_trks:
            person = persons_list[person_idx]
            person.misses += 1
            person.hits = 0
            person.predict_state(time.time() - start_time)

            x = person.x.T[0].tolist()
            x = [x[0] - int(person.height / 2), x[1] - int(person.width / 2),
                 x[0] + int(person.height / 2), x[1] + int(person.width / 2)]
            persons_boxes[person_idx] = x
            person.box = x

    # Create tracker for new detections
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = detections[idx]
            center = [z[0] + int((z[2] - z[0]) / 2),
                      z[1] + int((z[3] - z[1]) / 2)]
            center = np.expand_dims(center, axis=0).T

            person = tracker_center.Tracker()
            person.x = np.expand_dims([center[0], center[1], 0, 0], axis=0).T

            person.width = z[3] - z[1]
            person.height = z[2] - z[0]
            x = person.x.T[0].tolist()
            x = [x[0] - int(person.height / 2), x[1] - int(person.width / 2),
                 x[0] + int(person.height / 2), x[1] + int(person.width / 2)]
            person.box = x

            # Assign an ID for the tracker
            next_id += 1
            person.id = next_id

            persons_list.append(person)
            persons_boxes.append(x)

    # Draw good boxes
    for person in persons_list:
        if ((person.hits >= MIN_HITS) and (person.misses <= MAX_MISSES)):
            x = person.x.T[0].tolist()[0]
            y = person.x.T[0].tolist()[1]
            img = cv2.circle(img, (y, x), 4, (0, 255, 0), -1)
            img = tools.draw_box(person.id, img, person.box)

    # Remove lost trackers
    persons_list = [i for i in persons_list if i.misses <= MAX_MISSES]

    return img


def main():
    """
    Processes input video frame by frame, draws bounding boxes
    and saves the result, if needed
    """

    if len(sys.argv) != 2:
        print("Usage: python3 {} <path_to_video>".format(sys.argv[0]))
        return

    cap = cv2.VideoCapture(sys.argv[1])
    # cap = cv2.VideoCapture(0)

    if (SAVE_VIDEO):
        video_width = int(cap.get(3))
        video_height = int(cap.get(4))
        video_fps = int(cap.get(5))
        video_name = "../output/" + sys.argv[1].split('/')[-1]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(video_name, fourcc, video_fps,
                              (video_width, video_height))

    frame = 1
    while(True):
        ret, img = cap.read()
        start_time = time.time()

        if (img is None):
            break

        # img = cv2.resize(img, (0, 0), fx=SHOW_SCALE, fy=SHOW_SCALE)

        if (frame % DETECT_FREQ == 0 or frame == 1):
            # print("Tracking")
            result = track(img, start_time)
        else:
            # print("Predicting")
            result = predict(img, start_time)

        # Save frame
        if (SAVE_VIDEO):
            out.write(result)

        result = cv2.resize(result, (0, 0), fx=SHOW_SCALE, fy=SHOW_SCALE)

        cv2.imshow("frame", result)
        frame += 1
        # print("Frame:", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if (SAVE_VIDEO):
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # det = detector.Detector('models/ssd_resnet50_v1')
    # det = detector.Detector('models/faster_rcnn_resnet101_kitti_2018_01_28', 2)
    det = detector_yolo.Detector()
    main()
