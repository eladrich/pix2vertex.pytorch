from .reconstructor import Reconstructor
from .detector import Detector
from .utils import vis_net_result, vis_depth_interactive, vis_pcloud_interactive, vis_depth_matplotlib

reconstructor = None
def reconstruct(image=None, verbose=False):
    global reconstructor
    if reconstructor is None:
        reconstructor = Reconstructor()
    if image is None:
        import os
        from .constants import sample_image
        image = os.path.join(os.path.dirname(__file__), sample_image)
        print('No image specified, using {} as default input image'.format(image))
    return reconstructor.run(image, verbose)
