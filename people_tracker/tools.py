"""
Helper functions
"""
import numpy as np
import cv2
from math import sqrt


def eucledian(c1, c2):
    """
    Calculates eucledian distance between c1 and c2

    Args:
        c1 (list): point [x, y]
        c2 (list): point [x, y]

    Returns:
        float: eucledian distance between c1 and c2
    """
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return sqrt(dx**2 + dy**2)


def iou(b1, b2):
    """
    Calculates Intersection Over Union(IOU) between b1 and b2

    Args:
        b1 (list): bounding box
        b2 (list): bounding box

    Returns:
        float: IOU metric
    """
    w = np.maximum(0, (np.minimum(b1[2], b2[2]) - np.maximum(b1[0], b2[0])))
    h = np.maximum(0, (np.minimum(b1[3], b2[3]) - np.maximum(b1[1], b2[1])))
    intersection = w * h
    area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    if (abs(area_b1 + area_b2 - intersection) < 0.0001):
        return 0.0

    return float(intersection) / (area_b1 + area_b2 - intersection)


def draw_box(id, img, box, box_color=(0, 255, 0)):
    """
    Draws bounding box and ID on the image

    Args:
        id (int): id of person
        img (np array): image to draw on
        box (list): bounding box
        box_color (tuple, optional): color of bounding box

    Returns:
        np array: updated image with bounding box and ID
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_thickness = 2
    font_color = (0, 0, 0)
    l, t, r, b = box[1], box[0], box[3], box[2]

    # Draw the bounding box
    cv2.rectangle(img, (l, t), (r, b), box_color, 4)

    # Draw filled rectangle for ID
    cv2.rectangle(img, (l - 2, t - 35),
                  (r + 2, t), box_color, -1, 1)

    # Draw ID
    cv2.putText(img, str(id), (l, t - 5), font,
                font_size, font_color, font_thickness, cv2.LINE_AA)

    return img
