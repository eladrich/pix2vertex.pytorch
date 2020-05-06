import os
import dlib
import numpy as np
import math
import cv2
from .utils import download_url, extract_file


class Detector:
    def __init__(self, predictor_path=None):
        self.detector = dlib.get_frontal_face_detector()
        self.set_predictor(predictor_path)
    
    def set_predictor(self, predictor_path):
        if predictor_path is None:
            from .constants import predictor_file
            predictor_path = os.path.join(os.path.dirname(__file__), predictor_file)
            print('Loading default detector weights from {}'.format(predictor_path))
        if not os.path.exists(predictor_path):
            from .constants import predictor_url
            os.makedirs(os.path.dirname(predictor_path), exist_ok=True)
            print('\tDownloading weights from {}...'.format(predictor_url))
            download_url(predictor_url, save_path=os.path.dirname(predictor_path))
            print('\tExtracting weights...')
            extract_file(predictor_path + '.bz2', os.path.dirname(predictor_path))
            print('\tDone!')
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_and_crop(self, img, img_size=512):
        dets = self.detector(img, 1)  # Take a single detection
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
        dets = dets[0]
        shape = self.predictor(img, dets)
        points = shape.parts()

        pts = np.array([[p.x, p.y] for p in points])
        min_x = np.min(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_x = np.max(pts[:, 0])
        max_y = np.max(pts[:, 1])
        box_width = (max_x - min_x) * 1.2
        box_height = (max_y - min_y) * 1.2
        bbox = np.array([min_y - box_height * 0.3, min_x, box_height, box_width]).astype(np.int)

        img_crop = Detector.adjust_box_and_crop(img, bbox, crop_percent=150, img_size=img_size)
        # img_crop = img[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3], :]
        return img_crop

    @staticmethod
    def adjust_box_and_crop(img, bbox, crop_percent=100, img_size=None):
        w_ext = math.floor(bbox[2])
        h_ext = math.floor(bbox[3])
        bbox_center = np.round(np.array([bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]))
        max_ext = np.round(crop_percent / 100 * max(w_ext, h_ext) / 2)
        top = max(1, bbox_center[0] - max_ext)
        left = max(1, bbox_center[1] - max_ext)
        bottom = min(img.shape[0], bbox_center[0] + max_ext)
        right = min(img.shape[1], bbox_center[1] + max_ext)
        height = bottom - top
        width = right - left
        # make the frame as square as possible
        if height < width:
            diff = width - height
            top_pad = int(max(0, np.floor(diff / 2) - top + 1))
            top = max(1, top - np.floor(diff / 2))
            bottom_pad = int(max(0, bottom + np.ceil(diff / 2) - img.shape[0]))
            bottom = min(img.shape[0], bottom + np.ceil(diff / 2))
            left_pad = 0
            right_pad = 0
        else:
            diff = height - width
            left_pad = int(max(0, np.floor(diff / 2) - left + 1))
            left = max(1, left - np.floor(diff / 2))
            right_pad = int(max(0, right + np.ceil(diff / 2) - img.shape[1]))
            right = min(img.shape[1], right + np.ceil(diff / 2))
            top_pad = 0
            bottom_pad = 0

        # crop the image
        img_crop = img[int(top):int(bottom), int(left):int(right), :]
        # pad the image
        img_crop = np.pad(img_crop, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), 'constant')

        if img_size is not None:
            img_crop = cv2.resize(img_crop, (img_size, img_size))
        return img_crop
