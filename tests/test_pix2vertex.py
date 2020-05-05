import numpy as np
from imageio import imread
import pix2vertex as p2v

from unittest import TestCase

class TestPix2Vertex(TestCase):
    def test_reconstruct(self):
        image = imread('examples/sample.jpg')
        results = p2v.reconstruct(image)
        self.assertEqual(len(results), 2)
