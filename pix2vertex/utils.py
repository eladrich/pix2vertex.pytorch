import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from itertools import product
import struct

ASCII_FACET = """  facet normal  {face[0]:e}  {face[1]:e}  {face[2]:e}
    outer loop
      vertex    {face[3]:e}  {face[4]:e}  {face[5]:e}
      vertex    {face[6]:e}  {face[7]:e}  {face[8]:e}
      vertex    {face[9]:e}  {face[10]:e}  {face[11]:e}
    endloop
  endfacet"""

BINARY_HEADER = "80sI"
BINARY_FACET = "12fH"


# Saving to STL is based on https://github.com/thearn/stl_tools/

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


def _build_binary_stl(facets):
    """returns a string of binary binary data for the stl file"""

    lines = [struct.pack(BINARY_HEADER, b'Binary STL Writer', len(facets)), ]
    for facet in facets:
        facet = list(facet)
        facet.append(0)  # need to pad the end with a unsigned short byte
        lines.append(struct.pack(BINARY_FACET, *facet))
    return lines


def _build_ascii_stl(facets):
    """returns a list of ascii lines for the stl file """

    lines = ['solid ffd_geom', ]
    for facet in facets:
        lines.append(ASCII_FACET.format(face=facet))
    lines.append('endsolid ffd_geom')
    return lines


def writeSTL(facets, file_name, ascii=False):
    """writes an ASCII or binary STL file"""

    f = open(file_name, 'wb')
    if ascii:
        lines = _build_ascii_stl(facets)
        lines_ = "\n".join(lines).encode("UTF-8")
        f.write(lines_)
    else:
        data = _build_binary_stl(facets)
        data = b"".join(data)
        f.write(data)

    f.close()


def save2stl(A, fn, scale=1, mask_val=None, ascii=False,
              max_width=235.,
              max_depth=140.,
              max_height=150.,
              solid=False,
              rotate=True,
              min_thickness_percent=0.1):
    """
    Reads a numpy array, and outputs an STL file
    Inputs:
     A (ndarray) -  an 'm' by 'n' 2D numpy array
     fn (string) -  filename to use for STL file
    Optional input:
     scale (float)  -  scales the height (surface) of the
                       resulting STL mesh. Tune to match needs
     mask_val (float) - any element of the inputted array that is less
                        than this value will not be included in the mesh.
                        default renders all vertices (x > -inf for all float x)
     ascii (bool)  -  sets the STL format to ascii or binary (default)
     max_width, max_depth, max_height (floats) - maximum size of the stl
                                                object (in mm). Match this to
                                                the dimensions of a 3D printer
                                                platform
     solid (bool): sets whether to create a solid geometry (with sides and
                    a bottom) or not.
     min_thickness_percent (float) : when creating the solid bottom face, this
                                    multiplier sets the minimum thickness in
                                    the final geometry (shallowest interior
                                    point to bottom face), as a percentage of
                                    the thickness of the model computed up to
                                    that point.
    Returns: (None)
    """

    # Remove Nans, set their values as the minimal one
    A = A.copy()
    A[np.isnan(A)] = A[~np.isnan(A)].min()

    m, n = A.shape
    if n >= m and rotate:
        # rotate to best fit a printing platform
        A = np.rot90(A, k=3)
        m, n = n, m
    A = scale * (A - A.min())

    if not mask_val:
        mask_val = A.min()  # - 1.

    facets = []
    mask = np.zeros((m, n))
    print("Creating top mesh...")
    for i, k in product(range(m - 1), range(n - 1)):
        this_pt = np.array([i - m / 2., k - n / 2., A[i, k]])
        top_right = np.array([i - m / 2., k + 1 - n / 2., A[i, k + 1]])
        bottom_left = np.array([i + 1. - m / 2., k - n / 2., A[i + 1, k]])
        bottom_right = np.array(
            [i + 1. - m / 2., k + 1 - n / 2., A[i + 1, k + 1]])

        n1, n2 = np.zeros(3), np.zeros(3)
        is_a, is_b = False, False
        if (this_pt[-1] > mask_val and top_right[-1] > mask_val and
                bottom_left[-1] > mask_val):
            facet = np.concatenate([n1, top_right, this_pt, bottom_right])
            mask[i, k] = 1
            mask[i, k + 1] = 1
            mask[i + 1, k] = 1
            facets.append(facet)

        if (this_pt[-1] > mask_val and bottom_right[-1] > mask_val and
                bottom_left[-1] > mask_val):
            facet = np.concatenate(
                [n2, bottom_right, this_pt, bottom_left])
            facets.append(facet)
            mask[i, k] = 1
            mask[i + 1, k + 1] = 1
            mask[i + 1, k] = 1

    print('\t', len(facets), 'facets')
    facets = np.array(facets)

    if solid:
        print("Computed edges...")
        edge_mask = np.sum([roll2d(mask, (i, k))
                            for i, k in product([-1, 0, 1], repeat=2)],
                           axis=0)
        edge_mask[np.where(edge_mask == 9.)] = 0.
        edge_mask[np.where(edge_mask != 0.)] = 1.
        edge_mask[0::m - 1, :] = 1.
        edge_mask[:, 0::n - 1] = 1.
        X, Y = np.where(edge_mask == 1.)
        locs = zip(X - m / 2., Y - n / 2.)

        zvals = facets[:, 5::3]
        zmin, zthickness = zvals.min(), zvals.ptp()

        minval = zmin - min_thickness_percent * zthickness

        bottom = []
        print("Extending edges, creating bottom...")
        for i, facet in enumerate(facets):
            if (facet[3], facet[4]) in locs:
                facets[i][5] = minval
            if (facet[6], facet[7]) in locs:
                facets[i][8] = minval
            if (facet[9], facet[10]) in locs:
                facets[i][11] = minval
            this_bottom = np.concatenate(
                [facet[:3], facet[6:8], [minval], facet[3:5], [minval],
                 facet[9:11], [minval]])
            bottom.append(this_bottom)

        facets = np.concatenate([facets, bottom])

    xsize = facets[:, 3::3].ptp()
    if xsize > max_width:
        facets = facets * float(max_width) / xsize

    ysize = facets[:, 4::3].ptp()
    if ysize > max_depth:
        facets = facets * float(max_depth) / ysize

    zsize = facets[:, 5::3].ptp()
    if zsize > max_height:
        facets = facets * float(max_height) / zsize

    print('Writing STL...')
    writeSTL(facets, fn, ascii=ascii)
    print('Done!')
