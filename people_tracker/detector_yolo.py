"""
This file contains the YOLO Detector class that
is being used for people detection

Attributes:
    MIN_CONFIDENCE (float): Minimal detection confidence
    MIN_H (int): Minimal height of bounding box
    MIN_W (int): Minimal width of a bounding box
"""

import numpy as np
import cv2
import tensorflow as tf

from yolo.utils.misc_utils import parse_anchors, read_class_names
from yolo.utils.nms_utils import gpu_nms
from yolo.utils.data_aug import letterbox_resize

from yolo.model import yolov3

MIN_CONFIDENCE = 0.5
MIN_H = 10
MIN_W = 10


class Detector(object):

    """
    Detects people on given image and returns list of pixel bounding boxes

    Attributes:
        anchors (tensor): YOLO anchors
        classes (tensor): YOLO detected classes
        input_data (tf.placeholder): placeholder for image input
        num_class (int): number of classes
        people_boxes (list): Description
        sess (tf.Session): Tensorflow session
        yolo_model (yolov3): YOLO model
    """

    def __init__(self):
        """
        Initializes YOLO tracker
        """
        self.people_boxes = []

        self.anchors = parse_anchors('./yolo/data/yolo_anchors.txt')
        self.classes = read_class_names('./yolo/data/coco.names')
        self.num_class = len(self.classes)

        self.sess = tf.compat.v1.Session()
        self.input_data = tf.placeholder(
            tf.float32, [1, 416, 416, 3], name='input_data')

        self.yolo_model = yolov3(self.num_class, self.anchors)

        with tf.variable_scope('yolov3'):
            pred_feature_maps = self.yolo_model.forward(
                self.input_data, False)
        pred_boxes, pred_confs, pred_probs = self.yolo_model.predict(
            pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        self.boxes, self.scores, self.labels = gpu_nms(
            pred_boxes, pred_scores, self.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(self.sess, './yolo/data/darknet_weights/yolov3.ckpt')

    def get_detections(self, image):
        """
        Finds people on the image and returns their predicted bounding boxes

        Args:
            image (numpy array): input image

        Returns:
            list: list of pixel bounding boxes
        """

        # Prepare image for YOLO
        img, resize_ratio, dw, dh = letterbox_resize(image, 416, 416)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.0

        # Detect objects
        boxes, scores, labels = self.sess.run(
            [self.boxes, self.scores, self.labels], feed_dict={self.input_data: img})

        # Convert normalized(0-1) bounding boxes to pixes bounding boxes
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / resize_ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / resize_ratio

        # Extract person detections
        tmp_people_boxes = []
        for i in range(len(boxes)):
            if(labels[i] == 0 and scores[i] > MIN_CONFIDENCE):
                box = np.array([int(boxes[i][1]), int(
                    boxes[i][0]), int(boxes[i][3]), int(boxes[i][2])])
                if ((box[3] - box[1]) >= MIN_W and (box[2] - box[0]) >= MIN_H):
                    tmp_people_boxes.append(box)
        self.people_boxes = tmp_people_boxes

        return self.people_boxes
