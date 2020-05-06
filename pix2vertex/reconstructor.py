import os
import copy
import cv2
import numpy as np
import torch
from imageio import imread

from .models import pix2pix
from .detector import Detector

class Reconstructor:
    def __init__(self, weights_path=None, detector=Detector()):
        self.detector = detector
        self.unet = pix2pix.UNet()
        self.set_initial_weights(weights_path)
        self.unet.train()  # As in the original pix2pix, works as InstanceNormalization
    
    def set_initial_weights(self, weights_path):
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__),
                    '../weights/faces_hybrid_and_rotated_2.pth')
        if not os.path.exists(weights_path):
            from .utils import download_from_gdrive
            from .constants import p2v_model_gdrive_id
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            download_from_gdrive(p2v_model_gdrive_id, weights_path)
        self.initial_weights = torch.load(weights_path)

    def run(self, image, verbose=False):
        if type(image) is str:
            image = imread(image)
        image_cropped = self.detector.detect_and_crop(image)
        net_res = self.run_net(image_cropped)
        final_res = self.post_process(net_res)
        if verbose:
            from . import vis_depth_interactive
            vis_depth_interactive(final_res['Z_surface'])
        return final_res['Z_surface'], image_cropped
        
    def run_net(self, img):
        # Because is actually instance normalization need to copy weights each time
        self.unet.load_state_dict(copy.deepcopy(self.initial_weights), strict=True)

        # Forward
        input = torch.from_numpy(img.transpose()).float()
        input = input.unsqueeze(0)
        input = input.transpose(2, 3)
        input = input.div(255.0).mul(2).add(-1)
        output = self.unet(input)
        output = output.add(1).div(2).mul(255)

        # Post Processing
        im_both = output.squeeze(0).detach().numpy().transpose().swapaxes(0, 1).copy()
        im_pncc = im_both[:, :, 0:3]
        im_depth = im_both[:, :, 3:6]
        im_depth[np.logical_and(im_depth < 10, im_depth > -10)] = 0
        im_pncc[np.logical_and(im_pncc < 10, im_pncc > -10)] = 0

        return {'pnnc': im_pncc, 'depth': im_depth}

    def post_process(self, net_res):
        im_pncc = net_res['pnnc'].astype(np.float64)
        im_depth = net_res['depth'].astype(np.float64)
        net_X = im_depth[:, :, 0] * (1.3674) / 255 - 0.6852
        net_Y = im_depth[:, :, 1] * (1.8401) / 255 - 0.9035
        net_Z = im_depth[:, :, 2] * (0.7542) / 255 - 0.2997
        mask = np.any(im_depth, axis=2) * np.all(im_pncc, axis=2)

        X = np.tile(np.linspace(-1, 1, im_depth.shape[1]), (im_depth.shape[0], 1))
        Y = np.tile(np.linspace(1, -1, im_depth.shape[0]).reshape(-1, 1), (1, im_depth.shape[1]))

        # Normalize fixed grid according to the network result, as X,Y are actually redundant
        X = (X - np.mean(X[mask])) / np.std(X[mask]) * np.std(net_X[mask]) + np.mean(net_X[mask])
        Y = (Y - np.mean(Y[mask])) / np.std(Y[mask]) * np.std(net_Y[mask]) + np.mean(net_Y[mask])

        Z = net_Z * 2  # Due to image resizing

        f = 1 / (X[0, 1] - X[0, 0])

        Z_surface = Z * f
        Z_surface[mask == False] = np.nan
        Z[mask == False] = np.nan

        return {'Z': Z, 'X': X, 'Y': Y, 'Z_surface': Z_surface}
