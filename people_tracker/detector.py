"""
This file contains the Detector class that
is being used for people detection

Attributes:
    BOX_RATIO (float): Minimal H/W ratio
    MIN_CONFIDENCE (float): Minimal detection confidence
    MIN_H (int): Minimal height of bounding box
    MIN_W (int): Minimal width of a bounding box
"""
import numpy as np
import tensorflow as tf

BOX_RATIO = 0.05
MIN_CONFIDENCE = 0.25
MIN_H = 1
MIN_W = 1


class Detector(object):

    """
    Detects people on given image and returns list of pixel bounding boxes

    Attributes:
        boxes (tensor): part of the image with detected object
        classes (tensor): detected classes
        detection_graph (tf.Graph): Description
        image_tensor (tensor): Description
        num_detections (tensor): number of detections on a single image
        people_boxes (list): list of pixel bounding boxes
        scores (tensor): detection confidence (0-1)
        sess (tf.Session): tensorflow session
    """

    def __init__(self, model_name='models/ssd_mobilenet_v1_coco_2018_01_28', class_id=1):
        """
        Loads the given model

        Args:
            model_name (str, optional): name of the model's folder
        """
        self.people_boxes = []
        self.class_id = class_id

        model_path = model_name + '/frozen_inference_graph.pb'

        # Initialize tensorflow graph
        self.detection_graph = tf.Graph()

        # Configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            # Load model
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name(
                'image_tensor:0')

            # Box represents a part of the image with detected object
            self.boxes = self.detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Score represents detection confidence (0-1)
            self.scores = self.detection_graph.get_tensor_by_name(
                'detection_scores:0')
            # Classes represent detected classes. We only need Person class
            self.classes = self.detection_graph.get_tensor_by_name(
                'detection_classes:0')
            # Number of detections on a single image
            self.num_detections = self.detection_graph.get_tensor_by_name(
                'num_detections:0')

    def norm_to_pixel(self, box, dim):
        """
        Converts normalized(0-1) output bounding box of Neural Net
        to pixel bounding box

        Args:
            box (list): bounding box from NN
            dim (list): size of image

        Returns:
            numpy array: pixel bounding box
        """
        height, width = dim[0], dim[1]
        box_pixels = [int(box[0] * height), int(box[1] * width),
                      int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixels)

    def get_detections(self, image):
        """
        Finds people on the image and returns their predicted bounding boxes

        Args:
            image (numpy array): input image

        Returns:
            list: list of pixel bounding boxes
        """
        with self.detection_graph.as_default():
            nn_input = np.expand_dims(image, axis=0)

            # Detect objects
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: nn_input})

            # Extract detections as numpy arrays
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            # Extract person detections indices. Person has id = self.class_id
            people_ind = [i for i, v in enumerate(classes.tolist()) if(
                (v == self.class_id) and (scores[i] > MIN_CONFIDENCE))]

            # If there are people in the image
            if len(people_ind) != 0:
                tmp_people_boxes = []

                for i in people_ind:
                    box = self.norm_to_pixel(boxes[i], image.shape[0:2])
                    h = box[2] - box[0]
                    w = box[3] - box[1]
                    # If bounding box is not "strange"
                    if((h > MIN_H) and (w > MIN_W) and (h / w > BOX_RATIO)):
                        tmp_people_boxes.append(box)

                self.people_boxes = tmp_people_boxes

        return self.people_boxes
