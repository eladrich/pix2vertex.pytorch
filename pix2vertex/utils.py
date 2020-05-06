import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def vis_depth_matplotlib(img, Z, elevation=60, azimuth=45, stride=5):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Create X and Y data
    x = np.arange(0, 512, 1)
    y = np.arange(0, 512, 1)
    X, Y = np.meshgrid(x, y)

    ls = LightSource(azdeg=0, altdeg=90)
    # shade data, creating an rgb array.
    img_c = np.concatenate((img.astype('float') / 255, np.ones((512, 512, 1))), axis=2)
    rgb = ls.shade_rgb(img.astype('float') / 255, Z, blend_mode='overlay')

    surf = ax.plot_surface(X, Y, Z, cstride=stride, rstride=stride, linewidth=0, antialiased=False, facecolors=rgb)
    ax.view_init(elev=elevation, azim=azimuth)
    # ax.view_init(elev=90., azim=90)`
    # Show the plot
    plt.show()


def vis_depth_interactive(Z):
    import k3d
    Nx, Ny = 1, 1
    xmin, xmax = 0, Z.shape[0]
    ymin, ymax = 0, Z.shape[1]

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    x, y = np.meshgrid(x, y)
    plot = k3d.plot(grid_auto_fit=True, camera_auto_fit=False)
    plt_surface = k3d.surface(-Z.astype(np.float32), color=0xb2ccff,
                              bounds=[xmin, xmax, ymin, ymax])  # -Z for mirroring
    plot += plt_surface
    plot.display()
    plot.camera = [242.57934019166004, 267.50948550191197, -406.62328311352337, 256, 256, -8.300323486328125,
                   -0.13796270478729053, -0.987256298362836, -0.07931767413815752]
    return plot


def vis_pcloud_interactive(res, img):
    import k3d
    from colormap import rgb2hex
    color_vals = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color_vals[i, j] = int(rgb2hex(img[i, j, 0], img[i, j, 1], img[i, j, 2]).replace('#', '0x'), 0)
    colors = color_vals.flatten()

    points = np.stack((res['X'].flatten(), res['Y'].flatten(), res['Z'].flatten()), axis=1)

    invalid_inds = np.any(np.isnan(points), axis=1)
    points_valid = points[invalid_inds == False]
    colors_valid = colors[invalid_inds == False]
    plot = k3d.plot(grid_auto_fit=True, camera_auto_fit=False)
    plot += k3d.points(points_valid, colors_valid, point_size=0.01, compression_level=9, shader='flat')
    plot.display()
    plot.camera = [-0.3568942548181382, -0.12775125650240726, 3.5390732533009452, 0.33508163690567017,
                   0.3904658555984497, -0.0499117374420166, 0.11033077266672488, 0.9696364582197756, 0.2182481603445357]
    return plot


def vis_net_result(img, net_result):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.subplot(1, 3, 2)
    plt.imshow(net_result['pnnc'].astype(np.uint8))
    plt.title('PNCC Visualization')
    plt.subplot(1, 3, 3)
    plt.imshow(net_result['depth'][:, :, 2].astype(np.uint8), cmap='gray')
    plt.title('Depth Visualization')
    plt.show()


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, save_path):
    from six.moves import urllib
    save_path = os.path.expanduser(save_path)
    if not os.path.exists(save_path):
        makedir(save_path)

    filename = url.rpartition('/')[2]
    filepath = os.path.join(save_path, filename)

    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=url.split('/')[-1]) as t:  # all optional kwargs
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
            t.total = t.n
    except ValueError:
        raise Exception('Failed to download! Check URL: ' + url +
                        ' and local path: ' + save_path)


def extract_file(path, to_directory=None):
    path = os.path.expanduser(path)
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith(('.tar.gz', '.tgz')):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith(('tar.bz2', '.tbz')):
        opener, mode = tarfile.open, 'r:bz2'
    elif path.endswith('.bz2'):
        import bz2
        opener, mode = bz2.BZ2File, 'rb'
        with open(path[:-4], 'wb') as fp_out, opener(path, 'rb') as fp_in:
            for data in iter(lambda: fp_in.read(100 * 1024), b''):
                fp_out.write(data)
        return
    else:
        raise (ValueError,
               "Could not extract `{}` as no extractor is found!".format(path))

    if to_directory is None:
        to_directory = os.path.abspath(os.path.join(path, os.path.pardir))
    cwd = os.getcwd()
    os.chdir(to_directory)

    try:
        file = opener(path, mode)
        try:
            file.extractall()
        finally:
            file.close()
    finally:
        os.chdir(cwd)


def download_from_gdrive(id, destination):
    import requests
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    t = tqdm(unit='B', unit_scale=True, miniters=1)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                t.update(len(chunk))
                f.write(chunk)
