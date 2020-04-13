"""
Run people detection on input video

Attributes:
    VIDEO_SCALE (float): scale of the output video
"""

import cv2
import sys

import tools
import detector
import detector_yolo

# Output video scale
VIDEO_SCALE = 0.5


def detect(img):
    """
    Runs person detection on img

    Args:
        img (np.array): input image

    Returns:
        np.array: image with bounding boxes
    """
    detections = det.get_detections(img)

    for detection in detections:
        img = tools.draw_box(0, img, detection)

    return img


def main():
    """
    Opens video, feeds frames to the detector, and displays detections
    """
    if len(sys.argv) != 2:
        print("Usage: python3 {} <path_to_video>".format(sys.argv[0]))
        return

    cap = cv2.VideoCapture(sys.argv[1])
    while(True):
        ret, img = cap.read()

        if (img is None):
            break

        result = detect(img)

        result = cv2.resize(result, (0, 0), fx=VIDEO_SCALE, fy=VIDEO_SCALE)
        cv2.imshow("frame", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # det = detector.Detector('models/ssd_resnet50_v1')
    # det = detector.Detector('models/faster_rcnn')
    # det = detector.Detector()
    det = detector_yolo.Detector()

    main()
